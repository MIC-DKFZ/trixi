import numpy as np
from visdom import Visdom, _assert_opts, _opts2layout


class ExtraVisdom(Visdom):
    def histogram_3d(self, X, win=None, env=None, opts=None):
        """
        Given an array it plots the histrograms of the entries.

        Args:
            X : An array of at least 2 dimensions, where the first dimensions gives the number of histograms.
            win: Window name.
            env: Env name.
            opts: dict with options, especially opts['numbins'] (number of histogram bins) and opts['mutiplier']
        ( factor to stretch / queeze the values on the x axis) should be considered.

        Returns:
            The send result.
        """

        opts = {} if opts is None else opts
        _assert_opts(opts)

        X = np.asarray(X)
        assert X.ndim >= 2, 'X must have atleast 2 dimensions'

        opts['numbins'] = opts.get('numbins', min(30, len(X[0])))
        opts['mutiplier'] = opts.get('numbins', 100)

        traces = []
        for i, array in enumerate(X):
            array = array.flatten()
            bins, intervals = np.histogram(array, bins=opts['numbins'])

            x = []
            y = []
            z = []
            prev_interv = 0.
            for b, iv in zip(bins, intervals):
                interval_middle = float((prev_interv + iv) / 2.) * opts['mutiplier']
                z.append([float(b), float(b)])
                y.append([interval_middle, interval_middle])
                x.append([i * 2, i * 2 + 0.5])
                prev_interv = float(iv)
            traces.append(dict(
                z=z,
                x=x,
                y=y,
                # colorscale=[[i, 'rgb(%d,%d,255)' % (ci, ci)] for i in np.arange(0, 1.1, 0.1)],
                # autocolorscale=True,
                showscale=False,
                type='surface',
            ))

        return self._send({
            'data': traces,
            'win': win,
            'eid': env,
            'layout': _opts2layout(opts),
            'opts': opts,
        })
