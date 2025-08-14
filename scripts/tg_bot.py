from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from querying import QueryEngine

from config import *

qe = QueryEngine(
    model_path=model_path,
    vectorstore_path=vectorstore_path,
    embedding_model_name=embedding_model_name,
    device=device,
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    top_k=top_k
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь мне вопрос, и я постараюсь найти ответ на основе документов.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    await update.message.chat.send_action("typing")
    try:
        answer = qe.query(user_query, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k_gen=top_k_gen, typical_p=typical_p, repeat_penalty=repeat_penalty)
        await update.message.reply_text(answer or "Не удалось получить ответ.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

def main():
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.run_polling()

if __name__ == "__main__":
    main()
