require('dotenv').config();
const TelegramBot = require('node-telegram-bot-api');
const { setState, getState, clearState } = require('./state');
const { summarize, classifyBert, factCheck, searchSerpApi, analyzeWithAI } = require('./api');

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

// ── Helper ────────────────────────────────────────────────────

const MODEL_KEYBOARD = {
  reply_markup: {
    inline_keyboard: [[
      { text: 'BART', callback_data: 'BART' },
      { text: 'BERT', callback_data: 'BERT' }
    ]]
  }
};

function formatFactCheckResults(results) {
  if (!results || results.length === 0) {
    return 'Tidak ditemukan hasil fact-check terkait.';
  }
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
    'Selamat datang di *Hoax Checker Bot*!\n\nPilih model yang ingin digunakan:',
    { parse_mode: 'Markdown', ...MODEL_KEYBOARD }
  );
});

// ── Callback Query (GABUNGAN) ─────────────────────────────────

bot.on('callback_query', async (query) => {
  const chatId = query.message.chat.id;
  const data = query.data;
  console.log(`[callback] chatId=${chatId} data=${data}`);

  await bot.answerCallbackQuery(query.id);

  if (data === 'RESTART') {
    clearState(chatId);
    bot.sendMessage(chatId, 'Pilih model:', { ...MODEL_KEYBOARD });
    return;
  }

  if (data === 'BERT' || data === 'BART') {
    setState(chatId, { model: data, waitingInput: true });
    const desc = data === 'BART'
      ? 'BART akan meringkas teks kamu lalu mengecek fakta ke database fact-checker.'
      : 'BERT akan langsung mengklasifikasi apakah teks kamu hoax atau bukan.';
    bot.sendMessage(chatId,
      `Model dipilih: *${data}*\n${desc}\n\nSilakan kirim teks yang ingin dicek:`,
      { parse_mode: 'Markdown' }
    );
  }
});

bot.on('message', async (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;

  // Skip command
  if (!text || text.startsWith('/')) return;

  const state = getState(chatId);
  console.log(`[message] chatId=${chatId} model=${state.model} text="${text.slice(0, 50)}..."`);

  // Belum pilih model
  if (!state.model || !state.waitingInput) {
    bot.sendMessage(chatId, 'Silakan mulai dengan /start dan pilih model terlebih dahulu.');
    return;
  }

  // Set waitingInput false agar tidak proses ulang
  setState(chatId, { waitingInput: false });
  bot.sendMessage(chatId, 'Sedang memproses...');

  try {
    if (state.model === 'BART') {
      bot.sendMessage(chatId, 'Meringkas teks...');
      const summary = await summarize(text);

      bot.sendMessage(chatId,
        `*Ringkasan:*\n${summary}\n\nSedang mengecek fakta...`,
        { parse_mode: 'Markdown' }
      );

      const results = await factCheck(summary);

      if (results.length > 0) {
        // Ada hasil dari Google Fact Check API
        bot.sendMessage(chatId,
          `*Hasil Fact Check:*\n\n${formatFactCheckResults(results)}`,
          { parse_mode: 'Markdown', disable_web_page_preview: true }
        );
      } else {
        // Fallback ke SerpAPI
        bot.sendMessage(chatId, 'Tidak ada di database fact-check. Mencari via Google...');
        const serpResults = await searchSerpApi(summary);
        if (serpResults.length > 0) {
          bot.sendMessage(chatId, `Ditemukan ${serpResults.length} artikel terkait. Menganalisis dengan AI... 🤖`);
          const verdict = await analyzeWithAI(summary, serpResults);
          bot.sendMessage(chatId, verdict, { parse_mode: 'Markdown', disable_web_page_preview: true });
        } else {
          bot.sendMessage(chatId, 'Tidak ditemukan informasi terkait klaim ini.');
        }
      }

    } else if (state.model === 'BERT') {
      bot.sendMessage(chatId, 'Mengklasifikasi teks...');
      const result = await classifyBert(text);
      const emoji = result.label === 'BENAR' ? '✅' : '❌';
      bot.sendMessage(chatId,
        `${emoji} *Hasil Klasifikasi: ${result.label}*\nKepercayaan: ${result.confidence}%`,
        { parse_mode: 'Markdown' }
      );
    }

  } catch (err) {
    console.error('[error]', err.message);
    bot.sendMessage(chatId,
      'Terjadi kesalahan saat memproses. Pastikan backend sudah berjalan.\n\nKetik /start untuk mencoba lagi.'
    );
    clearState(chatId);
    return;
  }

  // Tawarkan cek lagi
  bot.sendMessage(chatId, 'Ingin mengecek teks lain?', {
    reply_markup: {
      inline_keyboard: [[
        { text: 'Ya, cek lagi', callback_data: state.model },
        { text: 'Ganti model', callback_data: 'RESTART' }
      ]]
    }
  });
});

console.log('Bot started...');
