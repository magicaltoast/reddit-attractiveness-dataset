import re
import regex

open_brace = "[({[]"
close_brace = "[)}\]]"
separator = "[\s|/\\-]*"
genders = f'(?:{"|".join(["f", "m", "ftm", "mtf", "t", "trans", "nb","male","female","man","woman"])})'

gender_block = f"(?:{open_brace}?(?P<gender>{genders}){close_brace}?)?"
age_block = f"{open_brace}?(?P<age>\s*\d{{2}}\s*){close_brace}?"
yeard_old_block = "(?P<years_old>y[\./]?[or]\.?|(?:years?)\s+old)"

age_gender_regex = re.compile(
    f"(?:^|\W){gender_block}{separator}{age_block}{yeard_old_block}?{separator}{gender_block.replace('gender','gender1')}(?:$|\W)",
    re.IGNORECASE,
)

just_gender_regex = re.compile(
    f"{open_brace}{separator}({genders}){separator}{close_brace}", re.IGNORECASE
)

ranking_re = regex.compile("(?:(\d)[,.-]?\s*)+")


def only_once(iterable):
    first = next(iterable, None)
    return first if next(iterable, None) is None else None


def extract_ranking(string):
    if rating := only_once(ranking_re.finditer(string)):
        res = []
        seen = set()

        for val in map(int, rating.captures(1)):
            if val not in seen:
                res.append(val)
                seen.add(val)
        return res


def extract_gender_age(string):
    matches = filter(
        lambda x: ((x[0] or x[3] or x[2]) and not (x[0] and x[2])),
        map(lambda x: x.groups(), age_gender_regex.finditer(string)),
    )

    if match := only_once(matches):
        gender1, age, _, gender2 = match
        gender = gender1 or gender2
        if gender:
            gender = gender.lower()
        return gender, int(age)

    elif match := only_once(just_gender_regex.finditer(string)):
        return match.group(1).lower(), None

    return None, None


regex = re.compile(
    "(?:^|\W)(?P<whole>\d+)(?:[,\.](?P<fraction>\d+))?(?:(?:\s*(?P<separator>[/\-~])\s*|\s+)(?:(?P<alt_whole>\d+)(?:[,\.](?P<alt_fraction>\d+))?))?(?:$|\W)"
)


def extract_rating(string):
    matches = regex.finditer(string)
    if match := only_once(matches):
        whole, frac, sep, alt_whole, alt_frac = match.groups()

        base = int(whole)

        if frac:
            base += int(frac) / (10 ** len(frac))

        if base > 10:
            return None

        if alt_whole:
            alt = int(alt_whole)

            if alt_frac:
                alt += int(alt_frac) / (10 ** len(alt_frac))

            if alt > 10:
                return None

            if sep == "/" and alt == 10:
                return base

            return (alt + base) / 2

        return base
