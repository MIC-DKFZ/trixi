try:
    from trixi.logger.message.telegrammessagelogger import TelegramMessageLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import telegram related modules:\n%s"
        % e.msg))
