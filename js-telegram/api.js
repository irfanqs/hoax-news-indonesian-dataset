const axios = require('axios');
require('dotenv').config();

const FACT_CHECK_URL = 'https://factchecktools.googleapis.com/v1alpha1/claims:search';
const SERP_API_URL = 'https://serpapi.com/search';

/**
 * Summarize teks panjang menggunakan mBART FastAPI
 */
async function summarize(text) {
  const res = await axios.post(`${process.env.BART_API_URL}/summarize`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' },
    timeout: 60000
  });
  return res.data.summary;
}

/**
 * Klasifikasi teks menggunakan IndoBERT FastAPI
 */
async function classifyBert(text) {
  const res = await axios.post(`${process.env.BERT_API_URL}/predict`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' },
    timeout: 60000
  });
  return res.data; // { label, confidence }
}

/**
 * Hit Google Fact Check API
 */
async function factCheck(query) {
  const res = await axios.get(FACT_CHECK_URL, {
    params: {
      query,
      key: process.env.FACT_CHECK_API_KEY,
      languageCode: 'id'
    },
    timeout: 10000
  });

  const claims = res.data.claims || [];
  return claims.slice(0, 3).map(claim => ({
    klaim: claim.text || '',
    rating: claim.claimReview?.[0]?.textualRating || '',
    sumber: claim.claimReview?.[0]?.publisher?.name || '',
    url: claim.claimReview?.[0]?.url || ''
  }));
}

/**
 * Fallback: Cari via SerpAPI jika Fact Check API tidak ada hasil
 */
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
    timeout: 10000
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

module.exports = { summarize, classifyBert, factCheck, searchSerpApi, analyzeWithAI };

/**
 * Analisis hasil scraping dengan Gemini AI
 * Merangkum bukti dan menilai apakah klaim adalah hoax
 */
async function analyzeWithAI(claim, serpResults) {
  const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

  // Bangun evidence dengan URL eksplisit
  const evidence = serpResults.map((r, i) =>
    `[${i + 1}] Judul: ${r.judul}\nIsi: ${r.snippet}\nURL: ${r.url}`
  ).join('\n\n');

  // Daftar URL untuk disertakan di sumber
  const sourceList = serpResults
    .filter(r => r.url)
    .map((r, i) => `${i + 1}. ${r.judul}\n   ${r.url}`)
    .join('\n');

  const prompt = `Kamu adalah sistem pendeteksi hoax berbahasa Indonesia yang akurat dan ringkas.

Klaim yang perlu dicek:
"${claim}"

Bukti dari ${serpResults.length} artikel hasil pencarian (beserta URL-nya):
${evidence}

Tugas kamu:
1. Tentukan apakah klaim ini HOAKS, BENAR, atau TIDAK DAPAT DIPASTIKAN
2. Berikan confidence score (0-100%) berdasarkan kekuatan bukti
3. Jelaskan alasan secara singkat dan terstruktur
4. Sertakan sumber dari URL yang diberikan di atas

Berikan output PERSIS dalam format berikut (jangan tambahkan teks lain di luar format):

[HOAKS/BENAR/TIDAK DAPAT DIPASTIKAN] Confidence: XX.X%

Klaim: "<tulis ulang klaim secara singkat>"

Alasan:
1. <alasan pertama berdasarkan bukti>
2. <alasan kedua berdasarkan bukti>
3. <alasan ketiga jika relevan>

Sumber:
${sourceList}

Gunakan bahasa Indonesia yang jelas dan mudah dipahami masyarakat awam.`;

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  let lastError;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await axios.post(GEMINI_URL, {
        contents: [{ parts: [{ text: prompt }] }]
      }, { timeout: 20000 });

      const text = res.data.candidates?.[0]?.content?.parts?.[0]?.text || '';
      return text.trim();
    } catch (err) {
      lastError = err;
      if (err.response?.status === 429) {
        const delay = attempt * 5000;
        console.log(`[gemini] 429 rate limit, retry ${attempt}/3 setelah ${delay / 1000}s...`);
        await sleep(delay);
      } else {
        throw err;
      }
    }
  }
  throw lastError;
}
