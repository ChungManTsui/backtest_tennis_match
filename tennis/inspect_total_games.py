import bz2, json, os

# For each TOTAL_GAMES file: get the last ltp before inPlay=True
# That's the closing pre-match price
base = r'c:\Users\trumen001\OneDrive - University of Surrey\Desktop\backtest_match\tennis\odd_data\2015\Aug'

results = []
count = 0

for root, dirs, files in os.walk(base):
    for f in files:
        if f.startswith('1.') and f.endswith('.bz2'):
            fp = os.path.join(root, f)
            try:
                with bz2.open(fp, 'rt') as fh:
                    first_line = fh.readline()
                d0 = json.loads(first_line)
                is_total = False
                event_id = None
                market_time = None
                runner_names = {}
                for mc in d0.get('mc', []):
                    md = mc.get('marketDefinition', {})
                    if md.get('marketType') == 'TOTAL_GAMES':
                        is_total = True
                        event_id = md.get('eventId')
                        market_time = md.get('marketTime')
                        for r in md.get('runners', []):
                            runner_names[r['id']] = r.get('name', '')
                if not is_total:
                    continue

                # Walk all lines, track ltp per runner, stop at inPlay
                ltp = {}
                pre_match_ltp = {}
                with bz2.open(fp, 'rt') as fh:
                    for line in fh:
                        d = json.loads(line)
                        for mc in d.get('mc', []):
                            md = mc.get('marketDefinition', {})
                            if md:
                                for r in md.get('runners', []):
                                    runner_names[r['id']] = r.get('name', '')
                                if md.get('inPlay') is True and not pre_match_ltp:
                                    pre_match_ltp = dict(ltp)
                            for rc in mc.get('rc', []):
                                rid = rc.get('id')
                                if rc.get('ltp'):
                                    ltp[rid] = rc['ltp']

                if not pre_match_ltp:
                    pre_match_ltp = dict(ltp)  # fallback: use whatever we have

                results.append({
                    'event_id': event_id,
                    'market_time': market_time,
                    'runner_names': runner_names,
                    'pre_match_ltp': pre_match_ltp,
                })
            except:
                pass
            count += 1

print(f'Processed {count} TOTAL_GAMES files in Aug 2015')
print(f'With prices: {sum(1 for r in results if r["pre_match_ltp"])}')
print('\nSample results:')
for r in results[:5]:
    print(f'  event={r["event_id"]} time={r["market_time"]}')
    for rid, name in r["runner_names"].items():
        price = r["pre_match_ltp"].get(rid, "N/A")
        print(f'    {name}: {price}')
