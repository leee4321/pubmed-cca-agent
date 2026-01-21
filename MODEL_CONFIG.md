# Model Configuration for PubMed CCA Agent

## Current Model Setup

### Results Section
- **Model**: `gemini-flash-latest`
- **Rationale**: Fast, cost-effective, suitable for structured statistical reporting
- **Output**: 1,500-2,500 words with detailed subsections

### Discussion Section  
- **Model**: `gemini-2.5-pro`
- **Rationale**: Superior reasoning and analytical capabilities for deep scientific insights
- **Output**: 3,000-4,000 words with minimal subsections (3-4 max)

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

### gemini-2.5-pro (Discussion)
- Speed: Slower (2-3x)
- Cost: $1.25/$5.00 per 1M tokens (input/output)
- RPM limit (free tier): 2
- Quality: Significantly better for complex analysis

## Expected Cost Per Run

### With New Configuration:
- Results (flash): ~$0.001 (3K input, 2.5K output)
- Discussion (pro): ~$0.040 (12K input, 5K output)
- **Total per run**: ~$0.041 (vs $0.003 with all-flash)

### Free Tier Limits:
- Can run ~60 times per day (limited by pro's 2 RPM)
- Still completely free if staying within limits

## Benefits of This Configuration

✅ **Best of both worlds**: Speed for Results, Quality for Discussion
✅ **Cost-effective**: Only use expensive model where it matters most
✅ **Better Discussion quality**: Deeper insights, better literature integration
✅ **Maintained speed**: Results generation still fast
✅ **Still free tier compatible**: 60 runs/day sufficient for most use cases
