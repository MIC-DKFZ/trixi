from vislogger import AbstractLogger
import telegram

class TelegramLogger(AbstractLogger):
    """
    Telegram logger, inherits the AbstractLogger and sends plots/logs to a chat via a Telegram bot.
    """

    def __init__(self, token, chat_id, **kwargs):
        """
        Creates a new TelegramLogger object.

        :param token: The token of the Telegram bot used.
        :type token: str
        :param chat_id: The chat ID for the chat between the user and the Telegram bot.
        :type chat_id: str
        """
        super(TelegramLogger, self).__init__(**kwargs)

        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=self.token)

    def show_text(self, text):
        """
        Sends a text to a chat using an existing Telegram bot.
        :param text: Text message to be sent to the bot.
        :type text: str
        """
        self.bot.send_message(chat_id=self.chat_id, text=text)

    def show_image(self, image_path):
        """
        Sends an image file to a chat using an existing Telegram bot.
        :param image_path: Path to the image file to be sent to the chat.
        :type image_path: str
        """
        self.bot.send_photo(chat_id=self.chat_id, photo=open(image_path, 'rb'))

    def show_value(self, value, counter=None, tag=None, **kwargs):
        """
        Sends a value to a chat using an existing Telegram bot.

        :param value: Value to be sent to the chat.
        :param counter: Optional counter to be sent in conjunction with the value.
        :param tag: Tag to be used for value.
        :type tag: str
        """
        if tag is None:
            tag = 'Value'

        message = tag+': ' + str(value)
        if counter is not None:
            message += ', Counter:' + str(counter)

        self.show_text(text=message)
