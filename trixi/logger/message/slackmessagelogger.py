import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from slackclient import SlackClient

from trixi.util.util import figure_to_image, get_image_as_buffered_file

from trixi.logger.plt.numpyseabornimageplotlogger import NumpySeabornImagePlotLogger


class SlackMessageLogger(NumpySeabornImagePlotLogger):
    """
    Slack logger, inherits the NumpySeabornImagePlotLogger and sends plots/logs to a chat via a Slack bot.
    """

    @staticmethod
    def find_uid_for_email(slack_client, email):
        """
        Returns the slack user id for a given email

        Args:
            slack_client: Slack client (already authorized)
            email: Workspace email address to get the user id for

        Returns:
            Slack workspace user id

        """

        user_list = slack_client.api_call("users.list").get('members', [])
        user_id = ""

        for user in user_list:
            if user.get('profile', {}).get('email', '') == email:
                user_id = user['id']
                break

        return user_id

    @staticmethod
    def find_cid_for_user(slack_client, uid):
        """
        Returns the chat/channel id for a direct message of the bot with the given User

        Args:
            slack_client: Slack client (already authorized)
            uid: User id of the user

        Returns:
            chat/channel id for a direct message
        """

        direct_conv = slack_client.api_call(
            "conversations.open",
            users=[uid]
        )

        c_id = direct_conv.get('channel', {}).get('id', None)

        return c_id

    def __init__(self, token, user_email, exp_name=None, *args, **kwargs):
        """
        Creates a new NumpySeabornImagePlotLogger object.

        Args:
            token (str): The Bot User OAuth Access Token of the slack bot used.
            user_email (str): The user email in the workspace for the chat between the user and the slack bot.
            exp_name: name of the experiment, always used as message prefix
        """
        super(SlackMessageLogger, self).__init__(**kwargs)

        self.token = token
        self.user_email = user_email
        self.slack_client = SlackClient(token)
        self.uid = self.find_uid_for_email(self.slack_client, self.user_email)
        self.cid = self.find_cid_for_user(self.slack_client, self.uid)
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

    def send_message(self, message="", file=None):
        if file is None:
            ret_val = self.slack_client.api_call(
                "chat.postMessage",
                channel=self.cid,
                text=message
            )
            return ret_val.get('ts', '')
        else:
            ret_val = self.slack_client.api_call(
                "files.upload",
                channels=self.cid,
                file=file,
                title=message
            )
            return ret_val.get('ts', '')

    def show_text(self, text, *args, **kwargs):
        """
        Sends a text to a chat using an existing Telegram bot.

        Args:
            text (str): Text message to be sent to the bot.
        """
        if self.exp_name is not None:
            text = self.exp_name + ":\n" + text
        try:
            self.send_message(message=text)
        except:
            print("Could not send text to telegram")

    def show_image(self, image, *args,  **kwargs):
        """
        Sends an image file to a chat using an existing Telegram bot.

        Args:
            image (str or np array): Path to the image file to be sent to the chat.
        """
        try:
            if isinstance(image, str):
                with open(image, 'rb') as img_file:
                    self.send_message(message=self.exp_name, file=img_file)
            elif isinstance(image, np.ndarray):
                buf = get_image_as_buffered_file(image)
                self.send_message(message=self.exp_name, file=buf)

        except:
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
            self.send_message(message=caption, file=buf)
        except:
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

        figure = super().show_value(value=value, name=name, counter=counter, tag=tag)
        buf = get_image_as_buffered_file(figure)
        try:
            self.send_message(message=caption, file=buf)
        except:
            print("Could not send plot to telegram")

    def show_barplot(self, array, name="barplot", *args, **kwargs):
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
            self.send_message(message=caption, file=buf)
        except:
            print("Could not send plot to telegram")

    def show_lineplot(self, y_vals, x_vals=None, name="lineplot", *args, **kwargs):
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

        figure = super().show_lineplot(y_vals, x_vals, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)
        try:
            self.send_message(message=caption, file=buf)
        except:
            print("Could not send plot to telegram")

    def show_scatterplot(self, array, name="scatterplot", *args, **kwargs):
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
            self.send_message(message=caption, file=buf)
        except:
            print("Could not send plot to telegram")

    def show_piechart(self, array, name="piechart", *args, **kwargs):
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

        figure = super().show_piechart(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            self.send_message(message=caption, file=buf)
        except:
            print("Could not send plot to telegram")

    def print(self, text, *args, **kwargs):
        """Just calls show_text()"""
        self.show_text(text, *args, **kwargs)
