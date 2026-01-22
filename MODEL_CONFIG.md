# Model Configuration for PubMed CCA Agent

## Current Model Setup

### Results Section
- **Model**: `gemini-flash-latest`
- **Rationale**: Fast, cost-effective, suitable for structured statistical reporting
- **Output**: 1,500-2,500 words with detailed subsections

### Discussion Section  
- **Model**: `gemini-2.5-flash`
- **Rationale**: Latest generation Flash model with improved reasoning capabilities
- **Output**: 3,000-4,000 words with minimal subsections (3-4 max)
- **Note**: Originally planned to use `gemini-2.5-pro`, but it's not available in free tier (quota: 0)

## Discussion Section Improvements

### Key Changes:
1. **Reduced Subsections**: Maximum 3-4 subsections (vs previous 8-9)
   - Opening (no heading)
   - "Genetic architecture and brain network substrates"
   - "Implications and future directions"
   - Closing (no heading)

2. **Enhanced Literature Integration**:
   - Target: 20-30+ citations throughout
   - 2-4 citations per paragraph in main discussion
   - Natural integration rather than isolated references

3. **Deeper Scientific Insights**:
   - Mechanistic models and testable hypotheses
   - Multi-level integration (molecular → cellular → circuit → behavioral)
   - Evolutionary and developmental perspectives
   - Alternative interpretations addressed

4. **Improved Narrative Flow**:
   - Themes integrated across paragraphs
   - Progressive argument building
   - Smooth transitions between ideas
   - Less rigid structure, more natural flow

## API Usage Comparison

### gemini-flash-latest (Results)
- Speed: Very fast
- Cost: $0.075/$0.30 per 1M tokens (input/output)
- RPM limit (free tier): 15

### gemini-2.5-flash (Discussion)
- Speed: Fast (similar to flash-latest, possibly slightly slower)
- Cost: Similar to flash-latest (~$0.075/$0.30 per 1M tokens)
- RPM limit (free tier): 15
- Quality: Latest generation with improved reasoning over previous flash models

### gemini-2.5-pro (Not Available in Free Tier)
- Status: ❌ Quota = 0 in free tier
- Would require paid tier subscription
- Cost on paid tier: $1.25/$5.00 per 1M tokens (input/output)

## Expected Cost Per Run

### With Current Configuration:
- Results (flash-latest): ~$0.001 (3K input, 2.5K output)
- Discussion (2.5-flash): ~$0.002 (12K input, 5K output)
- **Total per run**: ~$0.003 (all free tier compatible)

### Free Tier Limits:
- Can run ~450 times per day (limited by 15 RPM for both models)
- Completely free within quota limits

## Benefits of This Configuration

✅ **Latest generation models**: Both using modern Gemini 2.x architecture
✅ **Improved Discussion quality**: Enhanced prompts + newer model = better insights
✅ **Cost-effective**: All free tier, no quota concerns
✅ **Fast execution**: Both models are flash variants, no speed penalty
✅ **Free tier compatible**: 450 runs/day sufficient for extensive use
✅ **Better literature integration**: Enhanced prompts drive 20-30+ citations
