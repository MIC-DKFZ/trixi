try:
    from trixi.logger.message.telegrammessagelogger import TelegramMessageLogger
except ImportError as e:
    print("Could not import telegram related modules.")
    print(e)
