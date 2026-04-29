const axios = require('axios');
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

function jaccardSimilarity(text1, text2) {
  const tokenize = t => new Set(
    t.toLowerCase().split(/\s+/).filter(w => w.length > 2)
  );
  const a = tokenize(text1);
  const b = tokenize(text2);
  const intersection = new Set([...a].filter(w => b.has(w)));
  const union = new Set([...a, ...b]);
  return union.size === 0 ? 0 : intersection.size / union.size;
}

function decisionFusion(serpResults, bertResult, threshold = 1, simThreshold = 0.08) {
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
    const similarities = serpResults.map(r =>
      jaccardSimilarity(summary, r.judul + ' ' + r.snippet)
    );
    const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;

    if (avgSim >= simThreshold) {
      // Topik sama → cek apakah ada angka yang dikarang
      const nerCheck = detectNumberMismatch(summary, serpResults);
      if (nerCheck.mismatch) {
        // Angka di summary tidak cocok dengan artikel → kemungkinan dilebih-lebihkan
        return {
          label: 'HOAKS',
          source: 'ner',
          E, T,
          diff: 0,
          avgSim: avgSim.toFixed(3),
          mismatchedNums: nerCheck.mismatchedNums,
          summaryNums: nerCheck.summaryNums,
          articleNums: nerCheck.articleNums
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

  if (diff >= threshold) {
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
    msg += `_Topik cocok tapi angka tidak sesuai artikel (similarity: ${fusion.avgSim})_\n`;
    msg += `_Angka mencurigakan: ${fusion.mismatchedNums?.join(', ')} — artikel menyebut: ${fusion.articleNums?.join(', ')}_\n`;
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

module.exports = { summarize, classifyBert, factCheck, searchSerpApi, formatResult, decisionFusion };
