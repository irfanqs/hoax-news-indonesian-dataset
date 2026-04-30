const axios = require('axios');
const natural = require('natural');
const TfIdf = natural.TfIdf;
require('dotenv').config();

const FACT_CHECK_URL = 'https://factchecktools.googleapis.com/v1alpha1/claims:search';
const SERP_API_URL = 'https://serpapi.com/search';

// ── BART Summarize ────────────────────────────────────────────

async function summarize(text) {
  const res = await axios.post(`${process.env.BART_API_URL}/summarize`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' },
    timeout: 60000
  });
  return res.data.summary;
}

// ── BART Summarize Articles (batch) ──────────────────────────

async function summarizeArticles(serpResults) {
  return await Promise.all(
    serpResults.map(async (r) => {
      const combined = `${r.judul}. ${r.snippet}`;
      try {
        // Snippet sudah pendek, cukup gabung judul+snippet sebagai "summary"-nya
        // Hanya panggil BART jika teks cukup panjang (> 100 karakter)
        if (combined.length > 100) {
          const summary = await summarize(combined);
          return { ...r, summarized: summary };
        }
        return { ...r, summarized: combined };
      } catch {
        return { ...r, summarized: combined }; // fallback ke teks asli
      }
    })
  );
}

// ── BERT Classification ───────────────────────────────────────

async function classifyBert(text) {
  const res = await axios.post(`${process.env.BERT_API_URL}/predict`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' },
    timeout: 60000
  });
  return res.data; // { label, hoax_probability, valid_probability }
}

// ── Google Fact Check API ─────────────────────────────────────

async function factCheck(query) {
  const res = await axios.get(FACT_CHECK_URL, {
    params: {
      query,
      key: process.env.FACT_CHECK_API_KEY,
      languageCode: 'id'
    },
    timeout: 20000
  });

  const claims = res.data.claims || [];
  return claims.slice(0, 3).map(claim => ({
    klaim: claim.text || '',
    rating: claim.claimReview?.[0]?.textualRating || '',
    sumber: claim.claimReview?.[0]?.publisher?.name || '',
    url: claim.claimReview?.[0]?.url || ''
  }));
}

// ── SerpAPI Google Search ─────────────────────────────────────

async function searchSerpApi(query) {
  const res = await axios.get(SERP_API_URL, {
    params: {
      engine: 'google',
      q: `cek fakta ${query}`,
      api_key: process.env.SERPAPI_KEY,
      num: 5,
      hl: 'id',
      gl: 'id'
    },
    timeout: 15000
  });

  const organicResults = res.data.organic_results || [];
  return organicResults
    .filter(r => r.snippet)
    .slice(0, 3)
    .map(r => ({
      judul: r.title || '',
      snippet: r.snippet || '',
      url: r.link || ''
    }));
}

// ── Decision Fusion ───────────────────────────────────────────

const HOAX_KEYWORDS = [
  'hoaks', 'hoax', 'palsu', 'tidak benar', 'salah', 'menyesatkan',
  'disinformasi', 'klarifikasi', 'bohong', 'keliru', 'tidak terbukti', 'dibantah'
];
const FACT_KEYWORDS = [
  'benar', 'fakta', 'valid', 'resmi', 'terbukti', 'konfirmasi',
  'sahih', 'akurat', 'benar adanya', 'telah dikonfirmasi'
];

// ── Regex NER: Number Extraction ─────────────────────────────

function extractNumbers(text) {
  const matches = text.match(/\b\d[\d.,]*\b/g) || [];
  return matches
    .map(n => parseFloat(n.replace(/\./g, '').replace(',', '.')))
    .filter(n => !isNaN(n) && n >= 10 && (n < 1900 || n > 2100)); // abaikan tahun & angka kecil
}

function detectNumberMismatch(summary, serpResults) {
  const summaryNums = extractNumbers(summary);
  if (summaryNums.length === 0) return { mismatch: false, summaryNums: [], articleNums: [] };

  const allArticleText = serpResults.map(r => r.judul + ' ' + r.snippet).join(' ');
  const articleNums = extractNumbers(allArticleText);
  if (articleNums.length === 0) return { mismatch: false, summaryNums, articleNums };

  // Cek apakah tiap angka penting di summary punya padanan di artikel (toleransi 20%)
  const mismatched = summaryNums.filter(sNum => {
    const hasMatch = articleNums.some(aNum => {
      const ratio = sNum > aNum ? sNum / aNum : aNum / sNum;
      return ratio <= 1.2; // toleransi 20%
    });
    return !hasMatch;
  });

  return {
    mismatch: mismatched.length > 0,
    mismatchedNums: mismatched,
    summaryNums,
    articleNums
  };
}

function cosineSimilarity(text1, text2) {
  const tfidf = new TfIdf();
  tfidf.addDocument(text1.toLowerCase());
  tfidf.addDocument(text2.toLowerCase());

  // Kumpulkan semua term unik dari kedua dokumen
  const terms = new Set();
  tfidf.listTerms(0).forEach(t => terms.add(t.term));
  tfidf.listTerms(1).forEach(t => terms.add(t.term));

  // Buat vektor TF-IDF untuk masing-masing dokumen
  const vec1 = [], vec2 = [];
  for (const term of terms) {
    vec1.push(tfidf.tfidf(term, 0));
    vec2.push(tfidf.tfidf(term, 1));
  }

  // Hitung cosine similarity
  const dot = vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));

  return mag1 && mag2 ? dot / (mag1 * mag2) : 0;
}

async function nerCheck(summary, serpResults) {
  try {
    // Gunakan summarized jika tersedia untuk komparasi yang lebih konsisten
    const articles = serpResults.map(r => r.summarized || r.judul + ' ' + r.snippet);
    const res = await axios.post(`${process.env.BERT_API_URL}/ner`, { summary, articles }, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
      timeout: 15000
    });
    return res.data; // { summary_entities, article_entities, mismatch, has_mismatch }
  } catch {
    return null; // fallback ke regex jika endpoint tidak tersedia
  }
}

async function decisionFusion(serpResults, bertResult, threshold = 1, simThreshold = 0.15) {
  let E = 0;
  let T = 0;

  for (const r of serpResults) {
    const combined = (r.judul + ' ' + r.snippet).toLowerCase();
    const hasHoax = HOAX_KEYWORDS.some(k => combined.includes(k));
    const hasFact = FACT_KEYWORDS.some(k => combined.includes(k));

    if (hasHoax && !hasFact) E++;
    else if (hasFact && !hasHoax) T++;
  }

  // Jika tidak ada keyword match, gunakan similarity + NER number check
  if (E === 0 && T === 0 && serpResults.length > 0) {
    const summary = bertResult?._summary || '';
    // Gunakan field summarized jika tersedia, fallback ke judul+snippet
    const similarities = serpResults.map(r =>
      cosineSimilarity(summary, r.summarized || r.judul + ' ' + r.snippet)
    );
    const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;

    if (avgSim >= simThreshold) {
      // Topik sama → cek mismatch entitas via spaCy, fallback ke regex
      const spacyjResult = await nerCheck(summary, serpResults);
      const hasMismatch = spacyjResult
        ? spacyjResult.has_mismatch
        : detectNumberMismatch(summary, serpResults).mismatch;

      const mismatchDetail = spacyjResult
        ? Object.entries(spacyjResult.mismatch).map(([k, v]) => `${k}: ${v.join(', ')}`).join(' | ')
        : detectNumberMismatch(summary, serpResults).mismatchedNums?.join(', ');

      if (hasMismatch) {
        return {
          label: 'HOAKS',
          source: 'ner',
          E, T,
          diff: 0,
          avgSim: avgSim.toFixed(3),
          nerDetail: mismatchDetail
        };
      }
      T = serpResults.length;
    }

    return {
      label: T > 0 ? 'BENAR' : (bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN'),
      source: T > 0 ? 'similarity' : 'bert',
      E, T,
      diff: Math.abs(E - T),
      avgSim: avgSim.toFixed(3)
    };
  }

  const diff = Math.abs(E - T);
  const summary = bertResult?._summary || '';

  if (diff >= threshold) {
    // Keyword match cukup kuat → tetap cek NER sebelum memutuskan
    if (E <= T) {
      const spacyjResult = await nerCheck(summary, serpResults);
      const hasMismatch = spacyjResult
        ? spacyjResult.has_mismatch
        : detectNumberMismatch(summary, serpResults).mismatch;

      if (hasMismatch) {
        const mismatchDetail = spacyjResult
          ? Object.entries(spacyjResult.mismatch).map(([k, v]) => `${k}: ${v.join(', ')}`).join(' | ')
          : detectNumberMismatch(summary, serpResults).mismatchedNums?.join(', ');

        return { label: 'HOAKS', source: 'ner', E, T, diff, avgSim: null, nerDetail: mismatchDetail };
      }
    }
    return { label: E > T ? 'HOAKS' : 'BENAR', source: 'external', E, T, diff, avgSim: null };
  } else {
    return {
      label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
      source: 'bert',
      E, T, diff, avgSim: null
    };
  }
}

// ── Format hasil (Rule-Based, tanpa LLM) ─────────────────────

function stripMd(text) {
  return String(text)
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/_(.*?)_/g, '$1');
}

function formatResult(bertResult, factResults, serpResults, fusion = null) {
  const conf = bertResult.confidence != null ? parseFloat(bertResult.confidence) : null;
  const isHoaks = bertResult.label?.toUpperCase().includes('HOAKS') || bertResult.label?.toUpperCase().includes('HOAX');

  const hoaxPct = bertResult.hoax_probability != null
    ? (bertResult.hoax_probability * 100).toFixed(1)
    : conf != null
      ? (isHoaks ? conf.toFixed(1) : (100 - conf).toFixed(1))
      : null;

  const validPct = bertResult.valid_probability != null
    ? (bertResult.valid_probability * 100).toFixed(1)
    : conf != null
      ? (isHoaks ? (100 - conf).toFixed(1) : conf.toFixed(1))
      : null;

  const finalLabel = fusion ? fusion.label : bertResult.label;
  const fusionOverride = fusion && fusion.source === 'external' && fusion.label !== bertResult.label;

  let verdictEmoji, verdictText;
  if (finalLabel === 'HOAKS') {
    verdictEmoji = '⛔';
    verdictText = 'HOAKS';
  } else if (finalLabel === 'BENAR') {
    verdictEmoji = '✅';
    verdictText = 'KEMUNGKINAN FAKTA';
  } else {
    verdictEmoji = '❓';
    verdictText = 'TIDAK DAPAT DIPASTIKAN';
  }

  let msg = `🔍 *Hasil Analisis Berita*\n\n`;

  // Verdict utama di atas
  msg += `${verdictEmoji} *VERDICT: ${verdictText}*\n`;

  if (fusion && fusion.source === 'external') {
    msg += `_Diputuskan berdasarkan ${fusion.E + fusion.T} artikel eksternal_\n`;
  } else if (fusion && fusion.source === 'similarity') {
    msg += `_Diputuskan berdasarkan kemiripan topik artikel (similarity: ${fusion.avgSim})_\n`;
  } else if (fusion && fusion.source === 'ner') {
    msg += `_Topik cocok tapi entitas tidak sesuai artikel_\n`;
    if (fusion.nerDetail) msg += `_Entitas mencurigakan: ${fusion.nerDetail}_\n`;
  } else {
    msg += `_Diputuskan berdasarkan model BERT_\n`;
  }

  // Peringatan jika fusion berbeda dengan BERT
  if (fusionOverride) {
    msg += `\n⚠️ _Catatan: Model BERT mendeteksi ${bertResult.label} (${hoaxPct}%), namun ${fusion.E + fusion.T} artikel eksternal menunjukkan sebaliknya._\n`;
  }

  // Detail BERT
  msg += `\n${'─'.repeat(28)}\n`;
  msg += `📊 *Detail Model BERT:*\n`;
  msg += `├─ 🚨 Hoax  : *${hoaxPct ?? '?'}%*\n`;
  msg += `└─ ✅ Fakta : *${validPct ?? '?'}%*\n`;

  if (fusion) {
    msg += `\n📊 *Bukti Eksternal:*\n`;
    msg += `├─ 🚨 Artikel hoaks : ${fusion.E}\n`;
    msg += `└─ ✅ Artikel fakta : ${fusion.T}\n`;
  }

  if (factResults && factResults.length > 0) {
    msg += `\n${'─'.repeat(28)}\n`;
    msg += `🗂️ *Hasil Google Fact Check:*\n\n`;
    factResults.forEach((r, i) => {
      msg += `*${i + 1}.* ${stripMd(r.klaim)}\n`;
      msg += `   📌 Rating: ${stripMd(r.rating)}\n`;
      msg += `   🏛️ Sumber: ${stripMd(r.sumber)}\n`;
      if (r.url) msg += `   🔗 ${r.url}\n`;
      msg += '\n';
    });
  }

  if (serpResults && serpResults.length > 0) {
    msg += `${'─'.repeat(28)}\n`;
    msg += `📎 *Referensi dari Google:*\n\n`;
    serpResults.forEach((r, i) => {
      msg += `*${i + 1}.* ${stripMd(r.judul)}\n`;
      msg += `${stripMd(r.snippet)}\n`;
      msg += `🔗 ${r.url}\n\n`;
    });
  }

  if ((!factResults || factResults.length === 0) && (!serpResults || serpResults.length === 0)) {
    msg += `\n📭 _Tidak ditemukan referensi terkait._\n`;
  }

  return msg;
}

module.exports = { summarize, classifyBert, factCheck, searchSerpApi, summarizeArticles, formatResult, decisionFusion };
