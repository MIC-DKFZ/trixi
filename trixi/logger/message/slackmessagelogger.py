import numpy as np
import torch
import torchvision
from slack import WebClient

from trixi.util.util import get_image_as_buffered_file

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

        user_list = slack_client.users_list().get('members', [])
        user_id = ""

        for user in user_list:
            if user.get('profile', {}).get('email', '') == email:
                user_id = user['id']
                break

        return user_id

    @staticmethod
    def find_cid_for_user(slack_client, uid):
        """
        Returns the chat/channel id for a direct message of the bot with the given user

        Args:
            slack_client: Slack client (already authorized)
            uid: User id of the user

        Returns:
            chat/channel id for a direct message
        """

        direct_conv = slack_client.conversations_open(
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
        self.slack_client = WebClient(token)
        self.uid = self.find_uid_for_email(self.slack_client, self.user_email)
        self.cid = self.find_cid_for_user(self.slack_client, self.uid)
        self.exp_name = exp_name

        self.ts_dict = {}

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
        """
        Sends a message and a file if one is given

        Args:
            message: Message to be sent
            file:  File to be sent

        Returns:
            The timestamp (ts) of the message

        """
        if file is None:
            ret_val = self.slack_client.chat_postMessage(
                channel=self.cid,
                text=message
            )
            return ret_val.get('ts', '')
        else:
            ret_val = self.slack_client.files_upload(
                file=file,
                channels=self.cid,
                title=message
            )
            try:
                ts = list(ret_val['file']['shares']['private'].values())[0][0]['ts']
            except Exception as e:
                ts = ""
            return ts

    def delete_message(self, ts):
        """
        Deletes a direct message from the bot with the given timestamp (ts)

        Args:
            ts: Time stamp the message was sent

        """
        self.slack_client.chat_delete(
          channel=self.cid,
          ts=ts
        )

    def show_text(self, text, *args, **kwargs):
        """
        Sends a text to a chat using an existing slack bot.

        Args:
            text (str): Text message to be sent to the bot.
        """
        if self.exp_name is not None:
            text = self.exp_name + ":\n" + text
        try:
            self.send_message(message=text)
        except Exception as e:
            print("Could not send text to slack")

    def show_image(self, image, *args,  **kwargs):
        """
        Sends an image file to a chat using an existing slack bot.

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

        except Exception as e:
            print("Could not send image to slack")

    def show_image_grid(self, image_array, name=None, nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0, delete_last=True,
                        *args, **kwargs):
        """
        Sends an array of images to a chat using  an existing Slack bot. (Requires torch and torchvision)


        Args:
            image_array (np.narray / torch.tensor): Image array/tensor which will be sent as an image grid
            make_grid_kargs: Key word arguments for the torchvision make grid method
            delete_last: If a message with the same name was sent, delete it beforehand
        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        if isinstance(image_array, np.ndarray):
            image_array = torch.from_numpy(image_array)

        image_array = image_array.cpu()
        grid = torchvision.utils.make_grid(image_array, nrow=nrow, padding=padding, pad_value=pad_value,
                                           normalize=normalize, range=range, scale_each=scale_each)
        ndarr = grid.mul(255).clamp(0, 255).byte().numpy()
        buf = get_image_as_buffered_file(ndarr)
        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send image_grid to slack")

    def show_value(self, value, name, counter=None, tag=None, delete_last=True, *args, **kwargs):
        """
        Sends a value to a chat using an existing slack bot.

        Args:
            value: Value to be plotted sent to the chat.
            name: Name for the plot.
            counter: Optional counter to be sent in conjunction with the value.
            tag: Tag to be used as a label for the plot.
            delete_last: If a message with the same name was sent, delete it beforehand
        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        figure = super().show_value(value=value, name=name, counter=counter, tag=tag)
        buf = get_image_as_buffered_file(figure)
        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send plot to slack")

    def show_barplot(self, array, name="barplot", delete_last=True, *args, **kwargs):
        """
        Sends a barplot to a chat using an existing slack bot.

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure
            delete_last: If a message with the same name was sent, delete it beforehand

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        figure = super().show_barplot(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)
        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send plot to slack")

    def show_lineplot(self, y_vals, x_vals=None, name="lineplot", delete_last=True, *args, **kwargs):
        """
        Sends a lineplot to a chat using an existing slack bot.

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
                not set the points are assumed to be equally distributed in the interval [0, 1])
            name: The name of the figure
            delete_last: If a message with the same name was sent, delete it beforehand

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        figure = super().show_lineplot(y_vals, x_vals, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)
        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send plot to slack")

    def show_scatterplot(self, array, name="scatterplot", delete_last=True, *args, **kwargs):
        """
        Sends a scatterplot to a chat using an existing slack bot.

        Args:
            array: An array with size N x dim, where each element i \in N at X[i] results in a 2D
                (if dim = 2) or 3D (if dim = 3) point.
            name: The name of the figure
            delete_last: If a message with the same name was sent, delete it beforehand

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        figure = super().show_scatterplot(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send plot to slack")

    def show_piechart(self, array, name="piechart", delete_last=True, *args, **kwargs):
        """
        Sends a piechart to a chat using an existing slack bot.

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
                as the sum of all integers)
            name: The name of the figure
            delete_last: If a message with the same name was sent, delete it beforehand

        """

        caption = ""
        if self.exp_name is not None:
            caption += self.exp_name + "  "
        if name is not None:
            caption += name + "  "

        if delete_last and name in self.ts_dict:
            self.delete_message(self.ts_dict[name])

        figure = super().show_piechart(array, name, *args, **kwargs)
        buf = get_image_as_buffered_file(figure)

        try:
            ts = self.send_message(message=caption, file=buf)
            self.ts_dict[name] = ts
        except Exception as e:
            print("Could not send plot to slack")

    def print(self, text, *args, **kwargs):
        """Just calls :meth:`.show_text`"""
        self.show_text(text, *args, **kwargs)
