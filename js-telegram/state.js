const userState = new Map();

function setState(chatId, data) {
  const current = userState.get(chatId) || {};
  userState.set(chatId, { ...current, ...data });
}

function getState(chatId) {
  return userState.get(chatId) || {};
}

function clearState(chatId) {
  userState.delete(chatId);
}

module.exports = { setState, getState, clearState };
