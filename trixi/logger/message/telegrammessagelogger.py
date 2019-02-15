import numpy as np
import telegram
import torch
import torchvision
from trixi.util.util import get_image_as_buffered_file

from trixi.logger.plt.numpyseabornimageplotlogger import NumpySeabornImagePlotLogger


class TelegramMessageLogger(NumpySeabornImagePlotLogger):
    """
    Telegram logger, inherits the NumpySeabornImagePlotLogger and sends plots/logs to a chat via a Telegram bot.
    """

    def __init__(self, token, chat_id, exp_name=None, *args, **kwargs):
        """
        Creates a new TelegramMessageLogger object.

        Args:
            token (str): The token of the Telegram bot used.
            chat_id (str): The chat ID for the chat between the user and the Telegram bot.
        """
        super(TelegramMessageLogger, self).__init__(**kwargs)

        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=self.token)
        self.exp_name = exp_name

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert PyTorch variables and Tensors to numpy arrays
        """

        # convert args
        args = (a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args)

        # convert kwargs
        for key, data in kwargs.items():
            if torch.is_tensor(data):
                kwargs[key] = data.detach().cpu().numpy()

        return f(self, *args, **kwargs)

    def show_text(self, text, *args, **kwargs):
        """
        Sends a text to a chat using an existing Telegram bot.

        Args:
            text (str): Text message to be sent to the bot.
        """
        if self.exp_name is not None:
            text = self.exp_name + ":\n" + text
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            print("Could not send text to telegram")

    def show_image(self, image, *args, **kwargs):
        """
        Sends an image file to a chat using an existing Telegram bot.

        Args:
            image (str or np array): Path to the image file to be sent to the chat.
        """
        try:
            if isinstance(image, str):
                with open(image, 'rb') as img_file:
                    self.bot.send_photo(chat_id=self.chat_id, photo=img_file)
            elif isinstance(image, np.ndarray):
                buf = get_image_as_buffered_file(image)

                self.bot.send_photo(chat_id=self.chat_id, photo=buf)

        except Exception as e:
            print("Could not send image to telegram")

    def show_image_grid(self, image_array, name=None, nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0, *args, **kwargs):
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

        image_array = image_array.cpu()
        grid = torchvision.utils.make_grid(image_array, nrow=nrow, padding=padding, pad_value=pad_value,
                                           normalize=normalize, range=range, scale_each=scale_each)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        buf = get_image_as_buffered_file(ndarr)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send image_grid to telegram")

    def show_value(self, value, name, counter=None, tag=None, *args, **kwargs):
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

        figure = super().show_value(value, name, counter, tag, show=False)
        buf = get_image_as_buffered_file(figure)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send plot to telegram")

    def show_barplot(self, array, name=None, *args, **kwargs):
        """
        Sends a barplot to a chat using an existing Telegram bot.

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        figure = super().show_barplot(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send plot to telegram")

    def show_lineplot(self, y_vals, x_vals=None, name=None, *args, **kwargs):
        """
        Sends a lineplot to a chat using an existing Telegram bot.

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the figure

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        figure = super().show_lineplot(y_vals, x_vals, name, show=False, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send plot to telegram")

    def show_scatterplot(self, array, name=None, *args, **kwargs):
        """
        Sends a scatterplot to a chat using an existing Telegram bot.

        Args:
            array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
            3d (if dim = 3) point.
            name: The name of the figure

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        figure = super().show_scatterplot(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send plot to telegram")

    def show_piechart(self, array, name=None, *args, **kwargs):
        """
        Sends a piechart to a chat using an existing Telegram bot.

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
            as the sum of all integers)
            name: The name of the figure

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        figure = super().show_piechart(array, name,  *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            self.bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
        except Exception as e:
            print("Could not send plot to telegram")

    def print(self, text, *args, **kwargs):
        """Just calls show_text()"""
        self.show_text(text, *args, **kwargs)
