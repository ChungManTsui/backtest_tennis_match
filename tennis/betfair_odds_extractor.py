"""
betfair_odds_extractor.py
Extract total-games closing prices from Betfair bz2 data.

Handles two market formats:
  - TOTAL_GAMES (2015-2017): single line per market, runner names like "Over 22.5 Games"
  - COMBINED_TOTAL (2018+): multi-line market, runner hc field holds the line value

Strategy: read only the first 30 lines of the event-level bz2 (named <eventId>.bz2)
to find which events have relevant markets, then read only those market files.
Uses thread pool for parallel I/O.
"""
import bz2, json, os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

TOTAL_MARKET_TYPES = {'TOTAL_GAMES', 'COMBINED_TOTAL'}


def scan_event_file(event_bz2_path):
    """
    Read the event-level bz2 (first 30 lines) to get:
    - player names from MATCH_ODDS
    - market IDs for TOTAL_GAMES / COMBINED_TOTAL markets
    Returns (player_a, player_b, [(market_id, market_type), ...])
    """
    player_a = player_b = None
    total_markets = []
    try:
        with bz2.open(event_bz2_path, 'rt') as f:
            for i, raw in enumerate(f):
                if i > 30:
                    break
                d = json.loads(raw)
                for mc in d.get('mc', []):
                    md = mc.get('marketDefinition', {})
                    mt = md.get('marketType', '')
                    if mt == 'MATCH_ODDS' and player_a is None:
                        runners = md.get('runners', [])
                        if len(runners) == 2:
                            player_a = runners[0].get('name', '')
                            player_b = runners[1].get('name', '')
                    elif mt in TOTAL_MARKET_TYPES:
                        mid = mc.get('id', '')
                        if mid and (mid, mt) not in total_markets:
                            total_markets.append((mid, mt))
    except Exception:
        pass
    return player_a, player_b, total_markets


def extract_total_games_file(fp):
    """
    Read a TOTAL_GAMES market file (2015-2017 format).
    Runner names: "Over 22.5 Games" / "Under 22.5 Games"
    Returns (market_time, line_val, over_odds, under_odds) or None.
    """
    runner_names = {}
    ltp = {}
    pre_match_ltp = {}
    market_time = None

    try:
        with bz2.open(fp, 'rt') as f:
            for raw in f:
                d = json.loads(raw)
                for mc in d.get('mc', []):
                    md = mc.get('marketDefinition', {})
                    if md:
                        if not market_time:
                            market_time = md.get('marketTime', '')
                        for r in md.get('runners', []):
                            runner_names[r['id']] = r.get('name', '')
                        if md.get('inPlay') is True and not pre_match_ltp:
                            pre_match_ltp = dict(ltp)
                    for rc in mc.get('rc', []):
                        rid = rc.get('id')
                        if rc.get('ltp'):
                            ltp[rid] = rc['ltp']
    except Exception:
        return None

    if not pre_match_ltp:
        pre_match_ltp = dict(ltp)
    if not pre_match_ltp:
        return None

    over_rid = under_rid = None
    line_val = None
    for rid, name in runner_names.items():
        if name.startswith('Over'):
            over_rid = rid
            try:
                line_val = float(name.split()[1])
            except Exception:
                pass
        elif name.startswith('Under'):
            under_rid = rid

    if over_rid is None or under_rid is None or line_val is None:
        return None

    over_odds  = pre_match_ltp.get(over_rid)
    under_odds = pre_match_ltp.get(under_rid)
    if not over_odds or not under_odds:
        return None

    return market_time, line_val, over_odds, under_odds


def extract_combined_total_file(fp):
    """
    Read a COMBINED_TOTAL market file (2018+ format).
    Multiple lines in one market; rc entries have 'hc' field for the line.
    Picks the line where |over_odds - under_odds| is minimised (closest to 50/50).
    Returns (market_time, line_val, over_odds, under_odds) or None.
    """
    runner_hc = {}   # rid -> set of hc values
    over_rid = under_rid = None
    market_time = None

    # ltp_by_hc: hc -> {'over': ltp, 'under': ltp}
    ltp_by_hc = {}
    pre_match_ltp_by_hc = {}
    in_play = False

    try:
        with bz2.open(fp, 'rt') as f:
            for raw in f:
                d = json.loads(raw)
                for mc in d.get('mc', []):
                    md = mc.get('marketDefinition', {})
                    if md:
                        if not market_time:
                            market_time = md.get('marketTime', '')
                        for r in md.get('runners', []):
                            name = r.get('name', '')
                            rid  = r['id']
                            if name == 'Over':
                                over_rid = rid
                            elif name == 'Under':
                                under_rid = rid
                        if md.get('inPlay') is True and not pre_match_ltp_by_hc:
                            pre_match_ltp_by_hc = {k: dict(v) for k, v in ltp_by_hc.items()}
                    for rc in mc.get('rc', []):
                        rid = rc.get('id')
                        hc  = rc.get('hc')
                        ltp = rc.get('ltp')
                        if rid is None or hc is None or ltp is None:
                            continue
                        if hc not in ltp_by_hc:
                            ltp_by_hc[hc] = {}
                        if rid == over_rid:
                            ltp_by_hc[hc]['over'] = ltp
                        elif rid == under_rid:
                            ltp_by_hc[hc]['under'] = ltp
    except Exception:
        return None

    if not pre_match_ltp_by_hc:
        pre_match_ltp_by_hc = ltp_by_hc

    if not pre_match_ltp_by_hc or over_rid is None or under_rid is None:
        return None

    # Find the line with both over and under prices, closest to 50/50
    best_hc = None
    best_diff = float('inf')
    for hc, prices in pre_match_ltp_by_hc.items():
        if 'over' not in prices or 'under' not in prices:
            continue
        diff = abs(prices['over'] - prices['under'])
        if diff < best_diff:
            best_diff = diff
            best_hc = hc

    if best_hc is None:
        return None

    over_odds  = pre_match_ltp_by_hc[best_hc]['over']
    under_odds = pre_match_ltp_by_hc[best_hc]['under']
    return market_time, best_hc, over_odds, under_odds


def _process_event(event_dir, event_id, year, month, day):
    """Process one event folder. Returns list of records."""
    event_bz2 = os.path.join(event_dir, f'{event_id}.bz2')
    if not os.path.exists(event_bz2):
        return []

    player_a, player_b, total_markets = scan_event_file(event_bz2)
    if not total_markets or not player_a:
        return []

    records = []
    for mid, mtype in total_markets:
        market_fp = os.path.join(event_dir, f'{mid}.bz2')
        if not os.path.exists(market_fp):
            continue

        if mtype == 'TOTAL_GAMES':
            result = extract_total_games_file(market_fp)
        else:
            result = extract_combined_total_file(market_fp)

        if not result:
            continue
        market_time, line_val, over_odds, under_odds = result

        try:
            dt = datetime.fromisoformat(market_time.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            date_str = f'{year}-{month}-{day}'

        records.append({
            'date':       date_str,
            'player_a':   player_a,
            'player_b':   player_b,
            'line':       line_val,
            'over_odds':  over_odds,
            'under_odds': under_odds,
        })
    return records


def extract_year(odd_data_root, year, max_workers=8):
    year_dir = os.path.join(odd_data_root, str(year))
    if not os.path.isdir(year_dir):
        return []

    tasks = []
    for month in sorted(os.listdir(year_dir)):
        month_dir = os.path.join(year_dir, month)
        if not os.path.isdir(month_dir):
            continue
        for day in sorted(os.listdir(month_dir)):
            day_dir = os.path.join(month_dir, day)
            if not os.path.isdir(day_dir):
                continue
            for event_id in sorted(os.listdir(day_dir)):
                event_dir = os.path.join(day_dir, event_id)
                if os.path.isdir(event_dir):
                    tasks.append((event_dir, event_id, year, month, day))

    print(f'  {year}: {len(tasks)} event folders to scan...', flush=True)

    records = []
    events_with_total = 0
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_event, *t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result:
                events_with_total += 1
                records.extend(result)
            if done % 2000 == 0:
                print(f'    {done}/{len(tasks)} events processed, {len(records)} records so far...', flush=True)

    print(f'  {year}: {len(tasks)} events scanned, {events_with_total} with total-games market, {len(records)} records')
    return records


if __name__ == '__main__':
    odd_data_root = os.path.join(os.path.dirname(__file__), 'odd_data')
    out_path = os.path.join(os.path.dirname(__file__), 'data', 'betfair_total_games_atp.csv')

    years = [int(y) for y in sorted(os.listdir(odd_data_root)) if y.isdigit()]
    print(f'Extracting total-games odds for years: {years}')

    all_records = []
    for year in years:
        records = extract_year(odd_data_root, year)
        all_records.extend(records)

    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print(f'\nSaved {len(df)} records to {out_path}')
        print(df.head(10).to_string())
        print(f'\nLine distribution:\n{df["line"].value_counts().sort_index()}')
        print(f'\nOdds range: over {df["over_odds"].min():.2f}-{df["over_odds"].max():.2f}, '
              f'under {df["under_odds"].min():.2f}-{df["under_odds"].max():.2f}')
        print(f'\nDate range: {df["date"].min()} to {df["date"].max()}')
    else:
        print('No records found.')
