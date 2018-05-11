import io

import matplotlib.pyplot as plt
import numpy as np
import telegram
import torch
import torchvision
from PIL import Image

from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger


class TelegramLogger(NumpySeabornPlotLogger):
    """
    Telegram logger, inherits the AbstractLogger and sends plots/logs to a chat via a Telegram bot.
    """

    def __init__(self, token, chat_id, exp_name=None, **kwargs):
        """
        Creates a new TelegramLogger object.

        Args:
            token (str): The token of the Telegram bot used.
            chat_id (str): The chat ID for the chat between the user and the Telegram bot.
        """
        super(TelegramLogger, self).__init__(**kwargs)

        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=self.token)
        self.exp_name = exp_name

    def show_text(self, text, **kwargs):
        """
        Sends a text to a chat using an existing Telegram bot.

        Args:
            text (str): Text message to be sent to the bot.
        """
        if self.exp_name is not None:
            text = self.exp_name + ":\n" + text
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text)
        except:
            print("Could not send text to telegram")

    def show_image(self, image_path, **kwargs):
        """
        Sends an image file to a chat using an existing Telegram bot.

        Args:
            image_path (str): Path to the image file to be sent to the chat.
        """
        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=open(image_path, 'rb'))
        except:
            print("Could not send image to telegram")

    def show_image_grid(self, image_array, name=None, nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0, **kwargs):
        """
        Sends an array of images to a chat using  an existing Telegram bot. (Requires torch and torchvision)


        Args:
            image_array (np.narray / torch.tensor): Image array/ tensor which will be sent as an image grid
            make_grid_kargs: Key word arguments for the torchvision make grid method
        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if isinstance(image_array, np.ndarray):
            image_array = torch.from_numpy(image_array)

        buf = io.BytesIO()
        image_array = image_array.cpu()
        grid = torchvision.utils.make_grid(image_array, nrow=nrow, padding=padding, pad_value=pad_value,
                                           normalize=normalize, range=range, scale_each=scale_each)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im.save(buf, format="png")
        buf.seek(0)
        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except:
            print("Could not send image_grid to telegram")

    def show_value(self, value, name, counter=None, tag=None, **kwargs):
        """
        Sends a value to a chat using an existing Telegram bot.

        Args:
            value: Value to be plotted sent to the chat.
            name: Name for the plot.
            counter: Optional counter to be sent in conjunction with the value.
            tag: Tag to be used as a label for the plot.
        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        buf = io.BytesIO()
        figure = NumpySeabornPlotLogger.show_value(self, value, name, counter, tag)
        figure.savefig(buf, format='png')
        buf.seek(0)
        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except:
            print("Could not send plot to telegram")
        plt.close(figure)

        def show_barplot(self, *args, **kwargs):
            pass

        def show_lineplot(self, *args, **kwargs):
            pass

        def show_scatterplot(self, *args, **kwargs):
            pass

        def show_piechart(self, *args, **kwargs):
            pass
