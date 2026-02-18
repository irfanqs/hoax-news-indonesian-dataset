require('dotenv').config();
const TelegramBot = require('node-telegram-bot-api');
const { setState, getState, clearState } = require('./state');
const { summarize, classifyBert, factCheck, searchSerpApi, analyzeWithAI } = require('./api');

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

// ── Helper ────────────────────────────────────────────────────

function formatFactCheckResults(results) {
  if (!results || results.length === 0) return null;
  return results.map((r, i) =>
    `${i + 1}. *${r.klaim}*\n   Rating: ${r.rating}\n   Sumber: ${r.sumber}\n   ${r.url}`
  ).join('\n\n');
}

// ── /start ────────────────────────────────────────────────────

bot.onText(/\/start/, (msg) => {
  const chatId = msg.chat.id;
  console.log(`[/start] chatId=${chatId} user=${msg.chat.username || msg.chat.first_name}`);
  clearState(chatId);
  bot.sendMessage(chatId,
    '*Hoax Checker Bot* 🔍\n\nKirimkan teks berita atau pesan yang ingin kamu cek kebenarannya.\n\nBot akan:\n1. Menganalisis secara semantik (BERT)\n2. Meringkas teks (BART)\n3. Mengecek fakta & memberikan verdict AI',
    { parse_mode: 'Markdown' }
  );
});

// ── Pesan teks biasa ──────────────────────────────────────────

bot.on('message', async (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;

  // Skip command
  if (!text || text.startsWith('/')) return;

  console.log(`[message] chatId=${chatId} text="${text.slice(0, 60)}..."`);

  bot.sendMessage(chatId, '⏳ Memproses teks kamu...');

  try {
    // ── Step 1: BERT - Klasifikasi Semantik ──────────────────
    bot.sendMessage(chatId, '🧠 *Step 1/3: Analisis Semantik (BERT)*', { parse_mode: 'Markdown' });

    let bertResult = null;
    try {
      bertResult = await classifyBert(text);
      const emoji = bertResult.label === 'BENAR' ? '✅' : '❌';
      bot.sendMessage(chatId,
        `${emoji} *Hasil Semantik BERT:* ${bertResult.label}\nKepercayaan: ${bertResult.confidence}%`,
        { parse_mode: 'Markdown' }
      );
    } catch (bertErr) {
      console.error('[bert] error:', bertErr.message);
      bot.sendMessage(chatId, '⚠️ BERT tidak tersedia, melanjutkan ke langkah berikutnya...');
    }

    // ── Step 2: BART - Summarize ─────────────────────────────
    bot.sendMessage(chatId, '📝 *Step 2/3: Meringkas Teks (BART)*', { parse_mode: 'Markdown' });

    let summary = text; // fallback: pakai teks asli jika BART gagal
    try {
      summary = await summarize(text);
      bot.sendMessage(chatId,
        `*Ringkasan:*\n${summary}`,
        { parse_mode: 'Markdown' }
      );
    } catch (bartErr) {
      console.error('[bart] error:', bartErr.message);
      bot.sendMessage(chatId, '⚠️ BART tidak tersedia, menggunakan teks asli untuk fact check...');
    }

    // ── Step 3: Fact Check + AI Verdict ──────────────────────
    bot.sendMessage(chatId, '🔎 *Step 3/3: Pengecekan Fakta*', { parse_mode: 'Markdown' });

    const factResults = await factCheck(summary);
    const formatted = formatFactCheckResults(factResults);

    if (formatted) {
      // Ada hasil dari Google Fact Check API
      bot.sendMessage(chatId,
        `*Hasil Fact Check:*\n\n${formatted}`,
        { parse_mode: 'Markdown', disable_web_page_preview: true }
      );
    } else {
      // Fallback ke SerpAPI + AI Verdict
      bot.sendMessage(chatId, 'Tidak ada di database fact-check. Mencari via Google...');
      const serpResults = await searchSerpApi(summary);

      if (serpResults.length > 0) {
        bot.sendMessage(chatId, `Ditemukan ${serpResults.length} artikel terkait. Menganalisis dengan AI... 🤖`);
        const verdict = await analyzeWithAI(summary, serpResults);
        bot.sendMessage(chatId, verdict, { parse_mode: 'Markdown', disable_web_page_preview: true });
      } else {
        bot.sendMessage(chatId, 'Tidak ditemukan informasi terkait klaim ini di internet.');
      }
    }

  } catch (err) {
    console.error('[error]', err.message);
    bot.sendMessage(chatId,
      '❌ Terjadi kesalahan saat memproses. Pastikan backend sudah berjalan.\n\nKetik /start untuk mencoba lagi.'
    );
    return;
  }

  // Tawarkan cek lagi
  bot.sendMessage(chatId, 'Ingin mengecek teks lain? Kirimkan teks berikutnya atau ketik /start.', {
    reply_markup: {
      inline_keyboard: [[
        { text: '🔄 Cek teks lain', callback_data: 'RESTART' }
      ]]
    }
  });
});

// ── Callback Query ────────────────────────────────────────────

bot.on('callback_query', async (query) => {
  const chatId = query.message.chat.id;
  console.log(`[callback] chatId=${chatId} data=${query.data}`);
  await bot.answerCallbackQuery(query.id);

  if (query.data === 'RESTART') {
    clearState(chatId);
    bot.sendMessage(chatId, 'Silakan kirimkan teks yang ingin dicek:');
  }
});

console.log('Bot started...');
