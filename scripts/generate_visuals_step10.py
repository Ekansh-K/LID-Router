import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('dark_background')
sns.set_context("talk")

# --- Donut chart ---
routing_distribution = {
    "Mode A (Whisper)": 528,
    "Mode B (MMS)": 253,
    "Mode C (Fallback)": 59
}
labels = list(routing_distribution.keys())
sizes = list(routing_distribution.values())
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
explode = (0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1E1E2E')
ax.set_facecolor('#1E1E2E')
wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=False, startangle=90, pctdistance=0.85,
    textprops={'color': "white", 'fontsize': 14, 'weight': 'bold'}
)
centre_circle = plt.Circle((0, 0), 0.70, fc='#1E1E2E')
fig.gca().add_artist(centre_circle)
ax.axis('equal')
plt.title("Agent Routing Distribution (Step 10 Phase 3 F0)", color='white', fontsize=18, weight='bold', pad=20)
plt.savefig('results/circular_routing_distribution.png', facecolor='#1E1E2E', edgecolor='none', dpi=300, bbox_inches='tight')
print("Saved circular_routing_distribution.png")

# --- Load Step 10 ---
with open('results/step10_phase3_f0.json', 'r') as f:
    step10 = json.load(f)

per_lang = step10.get("per_language", {})
langs = sorted(list(per_lang.keys()))
cer_step10 = [per_lang[l]["mean_cer"] for l in langs]
x = np.arange(len(langs))

# --- Per-language CER: Step 10 only, y-axis 0 to 1.0 ---
fig2, ax2 = plt.subplots(figsize=(18, 8), facecolor='#1E1E2E')
ax2.set_facecolor('#1E1E2E')
bars = ax2.bar(x, cer_step10, 0.6, label='Step 10 Agent', color='#4ECDC4', zorder=3)
for bar, val in zip(bars, cer_step10):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
             f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=8, weight='bold')
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('Character Error Rate (Lower is Better)', color='white', fontsize=13)
ax2.set_title('Per-Language CER: Step 10 LID-Router Agent', color='white', fontsize=18, weight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(langs, rotation=45, ha='right', color='white', fontsize=11)
ax2.tick_params(axis='y', colors='white')
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax2.legend(facecolor='#2A2A3C', edgecolor='none', labelcolor='white', fontsize=12)
ax2.grid(color='#444455', linestyle='--', linewidth=1, alpha=0.5, axis='y', zorder=0)
plt.tight_layout()
plt.savefig('results/step10_per_lang_cer.png', facecolor='#1E1E2E', edgecolor='none', dpi=300)
print("Saved step10_per_lang_cer.png")

# --- Per-language routing stacked bar ---
records = step10.get("records", [])
routing_per_lang = {l: {'A': 0, 'B': 0, 'C': 0} for l in langs}
for r in records:
    l = r.get("true_lang")
    if l in routing_per_lang:
        mode = r.get("routing_mode", "B")
        routing_per_lang[l][mode] += 1

mode_A = [routing_per_lang[l]['A'] for l in langs]
mode_B = [routing_per_lang[l]['B'] for l in langs]
mode_C = [routing_per_lang[l]['C'] for l in langs]

fig3, ax3 = plt.subplots(figsize=(18, 8), facecolor='#1E1E2E')
ax3.set_facecolor('#1E1E2E')
ax3.bar(langs, mode_A, label='Mode A (Whisper)', color='#FF6B6B')
ax3.bar(langs, mode_B, bottom=mode_A, label='Mode B (MMS)', color='#4ECDC4')
ax3.bar(langs, mode_C, bottom=np.array(mode_A)+np.array(mode_B), label='Mode C (Fallback)', color='#FFE66D')
ax3.set_ylabel('Number of Samples', color='white')
ax3.set_title('Step 10 Agent Routing Decisions per Language', color='white', fontsize=18, weight='bold')
ax3.tick_params(axis='x', rotation=45, colors='white')
ax3.tick_params(axis='y', colors='white')
ax3.legend(facecolor='#2A2A3C', edgecolor='none', labelcolor='white')
plt.tight_layout()
plt.savefig('results/step10_per_lang_routing.png', facecolor='#1E1E2E', edgecolor='none', dpi=300)
print("Saved step10_per_lang_routing.png")
