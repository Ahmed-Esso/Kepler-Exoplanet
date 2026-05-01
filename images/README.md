# 📸 Screenshots Directory

This folder should contain all screenshots and visualizations referenced in `README_APP.md`.

## Image Files to Add

Place screenshots with the following names:

1. **dashboard.png**
   - Main dashboard/overview of the Streamlit app
   - Size: ~1200x800px recommended

2. **model_comparison.png**
   - Side-by-side model performance metrics
   - Size: ~1200x600px recommended

3. **roc_curves.png**
   - ROC curves for all 4 models
   - Size: ~1000x800px recommended

4. **feature_correlation.png**
   - Feature correlation heatmap
   - Size: ~1000x900px recommended

5. **clustering.png**
   - K-means clustering visualization
   - Size: ~1200x700px recommended

6. **association_rules.png**
   - Network visualization of feature associations
   - Size: ~1200x800px recommended

7. **confusion_matrices.png**
   - Confusion matrices for each model
   - Size: ~1200x600px recommended

8. **elbow_curve.png**
   - Elbow method curve for optimal clusters
   - Size: ~800x600px recommended

## Tips

- Use PNG format for better quality
- Keep consistent styling and colors
- Include axis labels and legends
- Ensure readability at smaller sizes
- Use the galaxy/space theme colors from the app:
  - Primary: #FF007F (pink)
  - Secondary: #00F0FF (cyan)
  - Accent: #AA00FF (purple)
  - Warning: #FFEA00 (yellow)
  - Success: #39FF14 (green)
  - Info: #1A73E8 (blue)

## How to Export from Streamlit

```python
# Save plots directly from streamlit widgets
st.pyplot(fig, use_container_width=True)

# For Plotly figures:
fig.write_image("images/figure_name.png")

# For Matplotlib:
plt.savefig("images/figure_name.png", dpi=300, bbox_inches='tight')
```

## How to Export from Jupyter

```python
# For Matplotlib:
plt.savefig("../images/figure_name.png", dpi=300, bbox_inches='tight')

# For Plotly:
fig.write_image("../images/figure_name.png", width=1200, height=800)
```

---

**Note**: The README links to these files, so please use the exact filenames listed above.
