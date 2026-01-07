import time

def get_time_str():
    return time.asctime(time.localtime(time.time()))

def print_with_time(message, style_ch=None):
    if not style_ch:
        print(f'{message} [{get_time_str()}]')
    else:
        print(style_ch*30, f'{message} [{get_time_str()}]', style_ch*30)
