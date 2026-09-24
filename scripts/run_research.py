import argparse, json
from factor_alpha.pipeline import run_research
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["synthetic","live"],default="synthetic"); p.add_argument("--focus",default="SPY"); p.add_argument("--outdir",default="results")
a=p.parse_args(); print(json.dumps(run_research(a.mode,a.outdir,a.focus),indent=2))
