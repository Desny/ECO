import os
import signal

# 别的模块在导入utils模块时，只能导入AttrDict类中的方法，属性等
__all__ = ['AttrDict']


def _term(sig_num, addition):
    print('current pid is %s, group id is %s' % (os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
