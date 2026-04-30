require('dotenv').config();
const TelegramBot = require('node-telegram-bot-api');
const { setState, getState, clearState } = require('./state');
const { summarize, classifyBert, factCheck, searchSerpApi, summarizeArticles, formatResult, decisionFusion } = require('./api');

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

// ── Helper ────────────────────────────────────────────────────

function stripMarkdown(text) {
  return String(text)
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/_(.*?)_/g, '$1');
}


// ── /start ────────────────────────────────────────────────────

bot.onText(/\/start/, (msg) => {
  const chatId = msg.chat.id;
  console.log(`[/start] chatId=${chatId} user=${msg.chat.username || msg.chat.first_name}`);
  clearState(chatId);
  bot.sendMessage(chatId,
    '*Hoax Checker Bot* 🔍\n\nKirimkan teks berita atau pesan yang ingin kamu cek kebenarannya.\n\nBot akan:\n1. Menganalisis secara semantik (BERT)\n2. Meringkas teks (BART)\n3. Mengecek fakta & memberikan verdict',
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
        `${emoji} Hasil Semantik BERT: ${bertResult.label}\nKepercayaan: ${bertResult.confidence}%`
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
      bot.sendMessage(chatId, `Ringkasan:\n${stripMarkdown(summary)}`);
    } catch (bartErr) {
      console.error('[bart] error:', bartErr.message);
      bot.sendMessage(chatId, '⚠️ BART tidak tersedia, menggunakan teks asli untuk fact check...');
    }

    // ── Step 3: Fact Check + AI Verdict ──────────────────────
    bot.sendMessage(chatId, '🔎 *Step 3/3: Pengecekan Fakta*', { parse_mode: 'Markdown' });

    const factResults = await factCheck(summary);
    let serpResults = factResults.length === 0 ? await searchSerpApi(summary) : [];

    if (factResults.length === 0 && serpResults.length === 0) {
      bot.sendMessage(chatId, 'Tidak ditemukan informasi terkait klaim ini di internet.');
    } else {
      // Summarize artikel SerpAPI agar komparasi lebih konsisten
      if (serpResults.length > 0) {
        bot.sendMessage(chatId, '📰 Meringkas artikel referensi...');
        serpResults = await summarizeArticles(serpResults);
      }

      const fusion = serpResults.length > 0
        ? await decisionFusion(serpResults, { ...bertResult, _summary: summary })
        : null;

      const msg = formatResult(bertResult, factResults, serpResults, fusion);
      bot.sendMessage(chatId, msg, { parse_mode: 'Markdown', disable_web_page_preview: true });
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
