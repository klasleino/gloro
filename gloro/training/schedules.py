import numpy as np
import re


def _linear_interp(start, end, duration):
    return [
        start + (end - start) * i / duration for i in range(duration)
    ]

def _exp_interp(start, end, duration):
    B = np.log(end / start) / duration

    return [start * np.exp(B * i) for i in range(duration)]

def _log_interp(start, end, duration):
    return [
        np.log((np.exp(end) - np.exp(start)) / duration * i + np.exp(start))
        for i in range(duration)
    ]

def _no_interp(start, duration):
    return [start for _ in range(duration)]


def _parse_section(
    section, duration, is_start=False, is_end=False, interpolated=False
):
    if not (section.startswith('[') and section.endswith(']')):
        raise ValueError(f'Invalid section: {section}')

    section = section[1:-1]

    if not (is_start or is_end or ':' in section):
        raise ValueError(f'Invalid section contents: {section}')

    if is_start:
        time = 0

    elif is_end:
        time = duration

    if ':' in section:
        time, value = section.split(':')

        if time.endswith('%'):
            time = int(float(time[:-1]) / 100. * duration)
        else:
            time = int(time)

    else:
        value = section

    if interpolated:
        value = float(value)

    else:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                pass

    return time, value


class Schedule(object):

    def __init__(self, schedule_string, duration, base_value):
        if schedule_string == 'fixed':
            self._schedule = [base_value]

        elif schedule_string == 'linear':
            self._schedule = Schedule(
                '[0.]-[1.]', duration, base_value).as_list()

        elif schedule_string == 'linear_half':
            self._schedule = Schedule(
                '[0.]-[50%:1.]-[1.]', duration, base_value).as_list()

        elif schedule_string.startswith('linear_from_'):
            start = float(schedule_string.split('linear_from_')[1]) / base_value
            self._schedule = Schedule(
                f'[{start}]-[1.]', duration, base_value).as_list()

        elif schedule_string.startswith('linear_half_from_'):
            start = float(
                schedule_string.split('linear_half_from_')[1]) / base_value
            self._schedule = Schedule(
                f'[{start}]-[50%:1.]-[1.]', duration, base_value).as_list()

        elif schedule_string == 'logarithmic':
            self._schedule = Schedule(
                '[0.01]-log-[1.]', duration, base_value).as_list()

        elif schedule_string == 'logarithmic_half':
            self._schedule = Schedule(
                '[0.01]-log-[50%:1.]-[1.]', duration, base_value).as_list()

        elif schedule_string.startswith('decay_to_'):
            end = float(schedule_string.split('decay_to_')[1]) / base_value
            self._schedule = Schedule(
                f'[1.]-exp-[{end}]', duration, base_value).as_list()

        elif schedule_string.startswith('decay_after_half_to_'):
            end = float(
                schedule_string.split('decay_after_half_to_')[1]) / base_value
            self._schedule = Schedule(
                f'[1.]-[50%:1.]-exp-[{end}]', duration, base_value).as_list()

        elif schedule_string.startswith('decay_until_half_to_'):
            end = float(
                schedule_string.split('decay_until_half_to_')[1]) / base_value
            self._schedule = Schedule(
                f'[1.]-exp-[50%:{end}]-[{end}]', duration, base_value).as_list()

        else:
            transitions_and_sections = [
                (transition, section)
                for transition, section in
                zip(
                    re.split(r'\[.*?\]', schedule_string),
                    re.findall(r'\[.*?\]', schedule_string))
            ]

            n = len(transitions_and_sections)

            uses_interpolation = not np.all(
                np.array([
                    transition for transition, _ in transitions_and_sections
                ]) == '')

            _time = 0
            self._schedule = []
            for i in range(n - 1):
                start_section = transitions_and_sections[i][1]
                transition, end_section = transitions_and_sections[i+1]

                start_time, start_val = _parse_section(
                    start_section,
                    duration,
                    is_start=i == 0,
                    is_end=False,
                    interpolated=uses_interpolation)
                end_time, end_val = _parse_section(
                    end_section,
                    duration,
                    is_start=False,
                    is_end=i + 1 == n - 1,
                    interpolated=uses_interpolation)

                if not (_time <= start_time <= end_time):
                    raise ValueError('Invalid time specification')

                start_val *= base_value
                end_val *= base_value

                if transition == '-':
                    self._schedule += _linear_interp(
                        start_val, end_val, end_time - start_time)

                elif transition == '-exp-':
                    self._schedule += _exp_interp(
                        start_val, end_val, end_time - start_time)

                elif transition == '-log-':
                    self._schedule += _log_interp(
                        start_val, end_val, end_time - start_time)

                else:
                    self._schedule += _no_interp(
                        start_val, end_time - start_time)

                    if i + 1 == n - 1 and end_time < duration - 1:
                        self._schedule.append(end_val)
            
    def __getitem__(self, index):
        return self._schedule[min(index, len(self._schedule) - 1)]

    def __len__(self):
        return len(self._schedule)

    def as_list(self):
        return self._schedule
