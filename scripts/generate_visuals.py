import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib

# Set style
plt.style.use('dark_background')
sns.set_context("talk")

# Data from step10_phase3_f0.json
routing_distribution = {
    "Mode A (Whisper)": 528,
    "Mode B (MMS)": 253,
    "Mode C (Fallback)": 59
}

labels = list(routing_distribution.keys())
sizes = list(routing_distribution.values())
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
explode = (0.05, 0.05, 0.05)  

# Create Donut Chart
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1E1E2E')
ax.set_facecolor('#1E1E2E')

wedges, texts, autotexts = ax.pie(
    sizes, 
    explode=explode, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%',
    shadow=False, 
    startangle=90,
    pctdistance=0.85,
    textprops={'color': "white", 'fontsize': 14, 'weight': 'bold'}
)

# Draw circle for donut effect
centre_circle = plt.Circle((0,0),0.70,fc='#1E1E2E')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.title("Agent Routing Distribution (Step 10 Phase 3 F0)", color='white', fontsize=18, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/circular_routing_distribution.png', facecolor='#1E1E2E', edgecolor='none', dpi=300)
print("Saved donut chart to results/circular_routing_distribution.png")

# Now let's create a performance progression line chart
steps = ['B3 Static MMS', 'A6 Learned', 'Step 8 (Track 3)', 'Step 9 (Phase 1+2)', 'Step 10 (Phase 3)']
cer_values = [0.1322, 0.2110, 0.1219, 0.1134, 0.1140]
wer_values = [0.0, 0.5000, 0.3420, 0.3342, 0.3349] # B3 WER was N/A, put 0 but don't plot it

fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='#1E1E2E')
ax2.set_facecolor('#1E1E2E')

ax2.plot(steps, cer_values, marker='o', markersize=10, linewidth=3, color='#4ECDC4', label='CER (raw)')
#ax2.plot(steps[1:], wer_values[1:], marker='s', markersize=10, linewidth=3, color='#FF6B6B', label='WER (raw)')

ax2.set_title('Pipeline Performance Progression (CER)', color='white', fontsize=18, weight='bold', pad=20)
ax2.set_ylabel('Character Error Rate (Lower is Better)', color='white', fontsize=14)
ax2.grid(color='#444455', linestyle='--', linewidth=1, alpha=0.5)
ax2.tick_params(colors='white', labelsize=12)

# Annotate points
for i, txt in enumerate(cer_values):
    ax2.annotate(f"{txt:.4f}", (steps[i], cer_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=12, weight='bold')

plt.legend(facecolor='#2A2A3C', edgecolor='none', labelcolor='white')
plt.tight_layout()
plt.savefig('results/cer_progression.png', facecolor='#1E1E2E', edgecolor='none', dpi=300)
print("Saved progression chart to results/cer_progression.png")

