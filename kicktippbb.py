"""KickTipp BetBot
Automated kicktipp.de bet placement.

Places bets to the upcomming matchday.
Unless specified by parameter it places the bets on all prediction games of the account.

Usage:
    kicktippbb.py [ --get-login-token ]
    kicktippbb.py [ --list-predictors ]
    kicktippbb.py [--use-login-token <token> ] [--dry-run] [--override-bets] [--deadline <duration>] [--predictor <value>] [--matchday <value>] [COMMUNITY]...

Options:
    COMMUNITY                   Name of the prediction game community to place bets on,
                                one or more names can be specified.
                                If no community name is given all available communities will be considered.
    --get-login-token           Just login and print the login token string
                                for later use with '--use-login-token' option
    --use-login-token <token>   Perform bets without interactive login, use login token instead.
    --override-bets             Override already placed bets.
    --deadline <duration>       Only place bets on matches that start in <duration> from now.
                                The duration format is <number><unit[m,h,d]>, e.g. 10m,5h or 1d
    --list-predictors           Display a list of predictors available to be used with '--predictor' option
    --predictor <value>         A specific predictor name to be used during calculation
    --dry-run                   Dont place any bet just print out predicitons
    --matchday <value>          Choose a specific matchday in the range of 1 to 34 to place bets on                                
"""

import sys
import datetime
import getpass
import re

from docopt import docopt
from robobrowser import RoboBrowser

import predictors.base
from helper.deadline import is_before_dealine, timedelta_tostring
from helper.match import Match

URL_BASE = 'http://www.kicktipp.de'
URL_LOGIN = URL_BASE + '/info/profil/login'

DEADLINE_REGEX = re.compile('([1-9][0-9]*)(m|h|d)')


def login(browser: RoboBrowser):
    """Log into the user account by asking for username and password.
    If login succeded the login cookie token is returned
    """
    while True:
        username, password = get_credentials()
        perform_login(browser, username, password)
        if not logged_in(browser):
            print("Email or password incorrect. Please try again.\n")
        else:
            return browser.session.cookies['login']


def perform_login(browser: RoboBrowser, username: str, password: str):
    """
    Open the log in page then fill out the form and submit
    """
    browser.open(URL_LOGIN)
    form = browser.get_form()
    form['kennung'] = username
    form['passwort'] = password
    browser.submit_form(form)


def get_credentials():
    """
    Ask the user for the credentials.
    """
    username = input("Username: ")
    password = getpass.getpass(prompt='Password: ')
    return username, password


def logged_in(browser: RoboBrowser):
    """
    Returns true if we are still on the login page
    """
    login_div = browser.find('div', content="Login")
    return True if not login_div else False


def get_table_rows(soup):
    """
    Get all table rows from the first tbody element found in soup parameter
    """
    tbody = soup.find('tbody')
    return [tr.find_all('td') for tr in tbody.find_all('tr')]


def extract_quote_points(text):
    """Extracts three integers from a bonus string like '3 - 7 - 7'."""
    match = re.match(r'\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*', text)
    if match:
        return tuple(map(int, match.groups()))
    return None


def parse_match_rows(browser: RoboBrowser, community, matchday=None):
    """Parses matches and optionally bonus points, adapting to column shifts."""
    browser.open(get_tippabgabe_url(community, matchday))
    content = get_kicktipp_content(browser)
    rows = get_table_rows(content)

    matchtuple = []
    lastmatch = None

    for row in rows:
        if len(row) < 5:
            continue  # Skip broken or unexpected rows

        # Default column indices
        idx_inputs = 3
        idx_odds = 4

        # Try to detect bonus points in column 3
        quote_points = extract_quote_points(row[3].get_text())
        if quote_points:
            idx_inputs = 4
            idx_odds = 5
        else:
            quote_points = (0, 0, 0)

        # Extract tip fields using dynamic indices
        heimtipp = row[idx_inputs].find(
            'input', id=lambda x: x and x.endswith('_heimTipp'))
        gasttipp = row[idx_inputs].find(
            'input', id=lambda x: x and x.endswith('_gastTipp'))

        # Extract match info
        try:
            odds_raw = row[idx_odds].get_text()
            odds = [odd.strip().replace("Quote:", "") for odd in odds_raw.split("/")]
            match = Match(row[1].get_text(), row[2].get_text(), row[0].get_text(), odds[0], odds[1], odds[2])
        except Exception as e:
            print("Error parsing match row:", e)
            continue

        if not match.match_date and lastmatch:
            match.match_date = lastmatch.match_date

        lastmatch = match
        matchtuple.append((heimtipp, gasttipp, match, quote_points))

    return matchtuple

def get_tippabgabe_url(community, matchday = None):
    tippabgabeurl = URL_BASE + '/' + community + '/tippabgabe'
    if matchday is None:
        return tippabgabeurl
    else:
        matchday = int(matchday)
        if matchday < 1:
            raise IndexError("The matchday '{}' is not valid, use only >= 1!".format(matchday))
        return tippabgabeurl + '?&spieltagIndex={matchday}'.format(matchday=matchday)


def get_kicktipp_content(browser: RoboBrowser):
    """
    Get the content view area from the kicktipp page.
    """
    content = browser.find_all(id='kicktipp-content')
    if content[0]:
        return content[0]
    return None


def get_communities(browser: RoboBrowser, desired_communities: list):
    """
    Get a list of all communities of the user
    """
    browser.open(URL_BASE + '/info/profil/meinetipprunden')
    content = get_kicktipp_content(browser)
    links = content.find_all('a')
    def gethreftext(link): return link.get('href').replace("/", "")

    def is_community(link):
        hreftext = gethreftext(link)
        if hreftext == link.get_text().lower().replace(" ", "-"):
            return True
        else:
            linkdiv = link.find('div', {'class': "menu-title-mit-tippglocke"})
            return linkdiv and linkdiv.get_text() == hreftext
    community_list = [gethreftext(link)
                      for link in links if is_community(link)]
    if len(desired_communities) > 0:
        return intersection(community_list, desired_communities)
    return community_list


def intersection(a, b):
    i = [x for x in a if x in b]
    return i

def fetch_scoring_rules(community):
    """Fetches scoring rules from the kicktipp page."""
    browser = RoboBrowser(parser="html5lib")
    url = f"{URL_BASE}/{community}/spielregeln"
    browser.open(url)

    scoring_table = None

    try:
        # Iterate through all tables on the page
        for table in browser.find_all('table'):
            rows = table.find_all('tr')
            has_sieg_row = False
            has_unentschieden_row = False

            # Check if it's the scoring rules table
            for row in rows:
                cells = row.find_all('td')
                if cells and cells[0].get_text().strip() == 'Sieg':
                    has_sieg_row = True
                elif cells and cells[0].get_text().strip() == 'Unentschieden':
                    has_unentschieden_row = True

            if has_sieg_row and has_unentschieden_row:
                scoring_table = table

        # Extract points from the scoring table if found
        if scoring_table:
            rows = scoring_table.find_all('tr')
            win_row = None
            draw_row = None

            for row in rows:
                cells = row.find_all('td')
                if cells and cells[0].get_text().strip() == 'Sieg':
                    win_row = row
                elif cells and cells[0].get_text().strip() == 'Unentschieden':
                    draw_row = row

            if win_row and draw_row:
                win_points_cells = win_row.find_all('td')
                win_tendency_points = int(win_points_cells[1].get_text())
                win_goal_difference_points = int(win_points_cells[2].get_text())
                win_exact_score_points = int(win_points_cells[3].get_text())

                draw_points_cells = draw_row.find_all('td')
                draw_tendency_points = int(draw_points_cells[1].get_text())
                draw_exact_score_points = int(draw_points_cells[3].get_text())
            else:
                raise ValueError("Could not find 'Sieg' or 'Unentschieden' rows in the identified scoring table.")
        else:
             # Set default scoring points if the table is not found
            win_exact_score_points = 4
            win_goal_difference_points = 3
            win_tendency_points = 2
            draw_exact_score_points = 4
            draw_tendency_points = 2
        
        return win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points


    except Exception as e:
        print(f"Error fetching or parsing scoring rules or MIN/MAX table: {e}")
        # Set default scoring points
        win_exact_score_points = 4
        win_goal_difference_points = 3
        win_tendency_points = 2
        draw_exact_score_points = 4
        draw_tendency_points = 2
        return win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points
    # --- End of Parsing Logic ---

def place_bets(browser: RoboBrowser, communities: list, predictor, override=False, deadline=None, dryrun=False, matchday=None):
    """Place bets on all given communities."""
    for com in communities:
        print("Community: {0}".format(com))
        matches = parse_match_rows(browser, com, matchday)
        submitform = browser.get_form()
        win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points = fetch_scoring_rules(com)
        for field_hometeam, field_roadteam, match, quote_points in matches:
            if not field_hometeam or not field_roadteam:
                homebet, roadbet = predictor.predict(match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points)
                print("{0} - betting {1}:{2}".format(match, homebet, roadbet))
                print("{0} - no bets possible".format(match))
                continue

            input_hometeam_value = submitform[field_hometeam.attrs['name']].value
            input_roadteam_value = submitform[field_roadteam.attrs['name']].value
            if not override and (input_hometeam_value or input_roadteam_value):
                homebet, roadbet = predictor.predict(match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points)
                print("{0} - betting {1}:{2}".format(match, homebet, roadbet))
                print("{0} - skipped, already placed {1}:{2}".format(match,
                                                                     input_hometeam_value, input_roadteam_value))
                continue

            if deadline is not None:
                if not is_before_dealine(deadline, match.match_date):
                    homebet, roadbet = predictor.predict(match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points)
                    print("{0} - betting {1}:{2}".format(match, homebet, roadbet))
                    time_to_match = match.match_date - datetime.datetime.now()
                    print("{0} - not betting yet, due in {1}".format(match,
                                                                     timedelta_tostring(time_to_match)))
                    continue

            homebet, roadbet = predictor.predict(match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points)
            print("{0} - betting {1}:{2}".format(match, homebet, roadbet))
            submitform[field_hometeam.attrs['name']] = str(homebet)
            submitform[field_roadteam.attrs['name']] = str(roadbet)
        if not dryrun:
            browser.submit_form(submitform, submit='submitbutton')
        else:
            print("INFO: Dry run, no bets were placed")


def validate_arguments(arguments):
    if arguments['--deadline']:
        deadline_value = arguments['--deadline']

        if not re.match(DEADLINE_REGEX, deadline_value):
            exit("Invalid deadline value ({}), use <Number><Unit>, Unit=[m,h,d]".format(
                deadline_value))


def choose_predictor(predictor_param, predictors):
    if(predictor_param):
        if(predictor_param in predictors):
            predictor = predictors[predictor_param]()
        else:
            exit('Unknown predictor: {}'.format(predictor_param))
    else:
        # Just get the first predictor in the dict and instanciate it
        predictor = next(iter(predictors.values()))()
    print("Using predictor: "+type(predictor).__name__)
    return predictor


def main(arguments):
    browser = RoboBrowser(parser="html5lib")

    validate_arguments(arguments)
    predictors_ = predictors.base.get_predictors()

    # Log in to kicktipp and print out the login cookie value
    if arguments['--get-login-token']:
        token = login(browser)
        print(token)
        exit(0)

    # Just list the predictors at hand and exit
    if arguments['--list-predictors']:
        [print(key) for key in predictors_.keys()]
        exit(0)

    # Use login token pass by argument or let the caller log in right here
    if arguments['--use-login-token']:
        token = arguments['--use-login-token']
    else:
        token = login(browser)

    communities = arguments['COMMUNITY']
    # Just use the token for all interactions with the website
    browser.session.cookies['login'] = token

    # Which communities are considered, fail if no were found
    if not communities:
        communities = get_communities(browser, communities)

    if(len(communities) == 0):
        exit("No community found!?")

    # Which prediction method is used
    predictor_param = arguments['--predictor'] if '--predictor' in arguments else None
    predictor = choose_predictor(predictor_param, predictors_)

    # Place bets
    place_bets(browser, communities, predictor,
               override=arguments['--override-bets'], deadline=arguments['--deadline'], dryrun=arguments['--dry-run'], matchday=arguments['--matchday'])


if __name__ == '__main__':
    arguments = docopt(__doc__, version='KickTipp BetBot 1.0')
    main(arguments)
