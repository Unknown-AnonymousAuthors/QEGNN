import os
import csv
import time
import matplotlib.pyplot as plt
import IPython.display as display

def _mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def _use_svg_display():
    #return
    # !pip install matplotlib_inline
    try:
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    except:
        pass

def _set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def get_log_dir(base_dir):
    for i in range(99999):
        path = f'{base_dir}/training_log{i+1}'
        if not os.path.exists(path):
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            os.mkdir(path)
            return path

class Logger:
    def __init__(self, filename='training_status', dir_path=None):
        self.filename = filename
        self.dir_path = dir_path
    
    def set_dir(self, dir_path):
        self.dir_path = dir_path
    
    def log(self, message, data):
        t = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        message = f'{t} - ' + message + '\n'

        filename = self.filename + '.log'
        if self.dir_path is not None:
            filename = self.dir_path + '/' + filename
        f = open(filename, 'a')
        f.write(message)
        f.close()

        # .csv
        if data is None:
            return
        if not isinstance(data, (list, tuple)):
            data = [data]
        data = list(data)

        filename = self.filename + '.csv'
        if self.dir_path is not None:
            filename = self.dir_path + '/' + filename
        with open(filename, 'a') as f:
            cw = csv.writer(f)
            cw.writerow(data)


class Animator:
    """For plotting data in animation."""
    def __init__(self, title, dir_path=None,
                 xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.legend = legend
        _use_svg_display()
        self.title = title
        self.dir_path = dir_path
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: _set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, self.legend)
        self.X, self.Y, self.fmts = None, None, fmts
        display.clear_output(wait=True)

    def step(self, epoch, y):
        plt.show()
        x = epoch
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.axes[0].set_title(self.title)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        self.save(epoch)
    
    def set_legend(self, legend):
        if isinstance(legend, tuple):
            legend = list(legend)
        self.legend = legend
    
    def set_dir(self, dir_path):
        self.dir_path = dir_path

    def save(self, epoch):
        if epoch % 50 == 0:
            file_name = 'learning_curve.png'
            if self.dir_path is not None:
                file_name = self.dir_path + '/' + file_name
            self.fig.savefig(file_name, dpi=600)

if __name__ == '__main__':
    animator = Animator('zz')
    for epoch in range(10):
        loss = 50 - epoch * 2
        animator.step(epoch + 1, loss)