import datetime
import time
from contextlib import contextmanager

from logger import LOGGER, LogLevel

DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
MIN_TIMESTAMP = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
MAX_TIMESTAMP = (datetime.datetime(9999, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc) -
                 datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)).total_seconds()


@contextmanager
def timer(task_name=None, log_level=LogLevel.DEBUG.value):
    start_time = time.time()
    try:
        yield
    except Exception as e:
        LOGGER.exception('timer')
        raise e
    finally:
        msg = '[{}] cost: {}s'.format(task_name, round(time.time() - start_time, 3))
        if log_level == LogLevel.DEBUG.value:
            LOGGER.debug(msg)
        elif log_level == LogLevel.INFO.value:
            LOGGER.info(msg)
        elif log_level == LogLevel.WARNING.value:
            LOGGER.warning(msg)
        elif log_level == LogLevel.ERROR.value:
            LOGGER.error(msg)
        elif log_level == LogLevel.CRITICAL.value:
            LOGGER.critical(msg)
    return True


def timestamp2datetime(timestamp):
    # 解决2038问题
    base_time = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    utc_time = base_time + datetime.timedelta(seconds=timestamp)
    local_offset = -time.timezone
    local_time = utc_time.astimezone(datetime.timezone(datetime.timedelta(seconds=local_offset)))
    return local_time


def timestamp2str(timestamp, format_):
    try:
        return timestamp2datetime(timestamp).strftime(format_)
    except:
        LOGGER.exception('timestamp2str')
        return None


def str2datetime(time_str, format_):
    try:
        return datetime.datetime.strptime(time_str, format_)
    except:
        LOGGER.exception('str2datetime')
        return None


def datetime2timestamp(datetime_):
    try:
        return datetime_.timestamp()
    except:
        LOGGER.exception('datetime2timestamp')
        return None


def str2timestamp(time_str, format_):
    return datetime2timestamp(str2datetime(time_str, format_))


def sec2hms(sec):
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    hms = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
    return hms


def hms2sec(hms):
    try:
        hours, minutes, seconds = map(int, hms.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except:
        LOGGER.exception('hms2sec')
    return None


def get_weekday(timestamp=None):
    try:
        if timestamp is not None:
            weekday = timestamp2datetime(timestamp).weekday() + 1
        else:
            weekday = datetime.datetime.now().weekday() + 1
        return weekday
    except:
        LOGGER.exception('get_weekday')
    return None


def get_day_second(timestamp=None):
    try:
        if timestamp is not None:
            datetime_ = timestamp2datetime(timestamp)
        else:
            datetime_ = datetime.datetime.now()
        if datetime_.tzinfo is not None:
            datetime_ = datetime_.replace(tzinfo=None)
        day_start_time = datetime.datetime.combine(datetime_.date(), datetime.datetime.min.time())
        return (datetime_ - day_start_time).total_seconds()
    except:
        LOGGER.exception('get_day_second')
    return None
