"""
RPGW-Net Experiment Runner
===========================
Single script for all experiment stages.

Usage:
    # Stage 1: GAT baseline (no GW, clean, full-shot)
    python experiments/run.py --stage 1

    # Stage 2: GAT + GW alignment (clean, full-shot)
    python experiments/run.py --stage 2

    # Stage 3: Noise + few-shot + GW
    python experiments/run.py --stage 3 --snr 5 --shots 5

    # Custom
    python experiments/run.py --source 0 --target 3 --gw partial --snr 5 --shots 5 --epochs 100
"""

import argparse, math, os, sys, time, yaml
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from data.cached_dataset import get_cached_dataloaders
from rpgw.train import RPGWNet

# ========== Preset stages (v2: CNN+GAT architecture) ==========
STAGES = {
    "1": {"gw": "none",    "snr": "inf", "shots": None, "epochs": 150, "desc": "CNN+GAT baseline (no GW)"},
    "2": {"gw": "partial", "snr": "inf", "shots": None, "epochs": 150, "desc": "CNN+GAT + Partial GW"},
    "3": {"gw": "partial", "snr": 5,     "shots": 5,    "epochs": 200, "desc": "Noise(5dB) + 5-shot + GW"},
    "4": {"gw": "vanilla", "snr": "inf", "shots": None, "epochs": 150, "desc": "CNN+GAT + Vanilla GW"},
    "5": {"gw": "entropic","snr": "inf", "shots": None, "epochs": 150, "desc": "CNN+GAT + Entropic GW"},
    "6": {"gw": "fused",   "snr": "inf", "shots": None, "epochs": 150, "desc": "CNN+GAT + Fused GW"},
}


def log(msg, logfile=None):
    print(msg, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(msg + '\n')

def run(cfg, save_name):
    device = torch.device('cuda')
    gw = cfg['model']['gw']['type']
    snr = cfg['noise']['target_snr_db']
    shots = cfg['train'].get('n_shot')
    epochs = cfg['train']['epochs']
    os.makedirs('logs', exist_ok=True)
    logfile = f'logs/{save_name}.log'

    tag = f"gw={gw} snr={snr} shot={shots}"
    log(f'Experiment: {tag}', logfile)
    log(f'GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB', logfile)

    src, tgt = cfg['experiment'].get('source_load', 0), cfg['experiment'].get('target_load', 3)
    loaders = get_cached_dataloaders(
        source_load=src, target_load=tgt,
        batch_size=cfg['train']['batch_size'],
        n_shot=shots,
        noise_snr_db=float(snr) if snr != "inf" else float('inf'),
    )

    model = RPGWNet(cfg).to(device)
    log(f'Params: {sum(p.numel() for p in model.parameters()):,} | GW: {model.use_gw}', logfile)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                           weight_decay=cfg['train']['weight_decay'])
    ce = nn.CrossEntropyLoss()
    gw_target = cfg['train']['loss'].get('gw_weight', 0.3)
    pretrain_epochs = cfg['train'].get('pretrain_epochs', 50)
    gw_warmup = cfg['train'].get('gw_warmup_epochs', 20)

    # LR scheduler
    milestones = cfg['train'].get('lr_decay_milestones', [100, 150])
    gamma = cfg['train'].get('lr_decay_gamma', 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)

    best_acc, best_ep = 0.0, 0
    history = []
    t_total = time.time()

    for ep in range(1, epochs + 1):
        # ---- GW weight scheduling (DAGCN-style) ----
        if model.use_gw:
            if ep <= pretrain_epochs:
                gw_w = 0.0                                  # Pretrain: no GW
            elif ep <= pretrain_epochs + gw_warmup:
                # Sigmoid warmup: gradually increase from 0 to gw_target
                progress = (ep - pretrain_epochs) / gw_warmup
                gw_w = gw_target / (1 + math.exp(-10 * (progress - 0.5)))
            else:
                gw_w = gw_target
        else:
            gw_w = 0.0

        model.train()
        tl, ta, n = 0.0, 0.0, 0
        for sb, tb in zip(loaders['source_train'], loaders['target_train']):
            out = model(sb, tb, 'train')
            loss = ce(out['logits'], out['labels'])
            if model.use_gw and gw_w > 0:
                loss = loss + gw_w * out['gw_loss']
            opt.zero_grad(); loss.backward(); opt.step()
            ta += (out['logits'].argmax(1) == out['labels']).sum().item()
            tl += loss.item() * len(out['labels']); n += len(out['labels'])

        scheduler.step()

        # Eval target
        model.eval()
        tacc, tn = 0.0, 0
        with torch.no_grad():
            for sb, tb in zip(loaders['source_test'], loaders['target_test']):
                out = model(sb, tb, 'eval')
                tacc += (out['tgt_logits'].argmax(1) == out['tgt_labels']).sum().item()
                tn += len(out['tgt_labels'])
        tacc /= max(tn, 1)
        history.append(tacc)
        if tacc > best_acc: best_acc, best_ep = tacc, ep

        if ep <= 3 or ep % 10 == 0:
            elapsed = time.time() - t_total
            eta = elapsed / ep * (epochs - ep)
            gw_str = f'GW_w {gw_w:.3f}' if model.use_gw else 'GW off'
            log(f'E{ep:3d} | Loss {tl/max(n,1):.4f} | Src {ta/max(n,1):.3f} | '
                  f'Tgt {tacc:.3f} | Best {best_acc:.3f}@{best_ep} | {gw_str} | {elapsed/60:.0f}m ETA {eta/60:.0f}m', logfile)

    total_m = (time.time() - t_total) / 60
    log(f'\n=== DONE {epochs}ep in {total_m:.1f}min ===', logfile)
    log(f'Best: epoch={best_ep} acc={best_acc:.4f}', logfile)
    hist_str = ', '.join([f'{x:.3f}' for x in history[-5:]])
    log(f'Last 5: [{hist_str}]', logfile)

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'epoch': best_ep, 'acc': best_acc, 'state': model.state_dict(),
                'config': cfg, 'history': history},
               f'checkpoints/{save_name}.pth')
    return best_acc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RPGW-Net Experiment Runner')
    p.add_argument('--stage', type=str, default=None, help='Preset stage (1-6)')
    p.add_argument('--source', type=int, default=0)
    p.add_argument('--target', type=int, default=3)
    p.add_argument('--gw', type=str, default='partial',
                   choices=['none', 'vanilla', 'entropic', 'fused', 'partial'])
    p.add_argument('--snr', type=float, default=float('inf'), help='Target SNR (inf=clean)')
    p.add_argument('--shots', type=int, default=None, help='Few-shot samples per class')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--name', type=str, default=None)
    args = p.parse_args()

    # Load base config
    with open('configs/default.yaml') as f:
        cfg = yaml.safe_load(f)

    # Apply stage preset or CLI args
    if args.stage and args.stage in STAGES:
        s = STAGES[args.stage]
        cfg['model']['gw']['type'] = s['gw']
        cfg['noise']['target_snr_db'] = s['snr']
        cfg['train']['n_shot'] = s['shots']
        cfg['train']['epochs'] = s['epochs']
        print(f'=== Stage {args.stage}: {s["desc"]} ===', flush=True)
    else:
        cfg['model']['gw']['type'] = args.gw
        cfg['noise']['target_snr_db'] = args.snr if args.snr != float('inf') else 'inf'
        cfg['train']['n_shot'] = args.shots
        cfg['train']['epochs'] = args.epochs

    cfg['train']['batch_size'] = args.batch_size
    cfg['train']['lr'] = args.lr
    cfg['experiment']['source_load'] = args.source
    cfg['experiment']['target_load'] = args.target

    gw_name = cfg['model']['gw']['type']
    snr_str = f'snr{int(args.snr)}' if args.snr != float('inf') else 'clean'
    shot_str = f'shot{args.shots}' if args.shots else 'full'
    save_name = args.name or f'rpgw_{gw_name}_{snr_str}_{shot_str}_s{args.source}t{args.target}'

    run(cfg, save_name)
