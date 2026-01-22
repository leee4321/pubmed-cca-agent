"""
Estimate Google API quota usage for one run of the PubMed CCA Agent.

This script analyzes the code to estimate:
1. Number of API calls
2. Approximate token usage
3. Quota consumption
"""

def estimate_api_usage():
    print("=" * 70)
    print("Google API Quota Usage Estimation for PubMed CCA Agent")
    print("=" * 70)
    
    # ========================================================================
    # 1. LLM API Calls (Google Gemini)
    # ========================================================================
    print("\n1. GEMINI LLM API CALLS:")
    print("-" * 70)
    
    # Results section generation: 1 call
    results_calls = 1
    print(f"   - Results section generation: {results_calls} call")
    
    # Discussion section generation: 1 call
    discussion_calls = 1
    print(f"   - Discussion section generation: {discussion_calls} call")
    
    total_llm_calls = results_calls + discussion_calls
    print(f"\n   TOTAL LLM API CALLS: {total_llm_calls}")
    
    # ========================================================================
    # 2. Token Usage Estimation
    # ========================================================================
    print("\n2. ESTIMATED TOKEN USAGE:")
    print("-" * 70)
    
    # Results section
    print("\n   Results Section:")
    results_input_tokens = 3000  # Prompt + data (CCA results, summary)
    results_output_tokens = 2500  # Target: 1,500-2,500 words ≈ 2,000-3,300 tokens
    print(f"   - Input tokens: ~{results_input_tokens:,}")
    print(f"   - Output tokens: ~{results_output_tokens:,}")
    results_total = results_input_tokens + results_output_tokens
    print(f"   - Total: ~{results_total:,} tokens")
    
    # Discussion section
    print("\n   Discussion Section:")
    discussion_input_tokens = 12000  # Prompt + findings + literature (8KB of abstracts)
    discussion_output_tokens = 4500  # Target: 2,500-3,500 words ≈ 3,300-4,700 tokens
    print(f"   - Input tokens: ~{discussion_input_tokens:,}")
    print(f"   - Output tokens: ~{discussion_output_tokens:,}")
    discussion_total = discussion_input_tokens + discussion_output_tokens
    print(f"   - Total: ~{discussion_total:,} tokens")
    
    total_tokens = results_total + discussion_total
    print(f"\n   TOTAL TOKENS: ~{total_tokens:,}")
    print(f"   - Input: ~{results_input_tokens + discussion_input_tokens:,}")
    print(f"   - Output: ~{results_output_tokens + discussion_output_tokens:,}")
    
    # ========================================================================
    # 3. PubMed API Calls (Free, no quota limit)
    # ========================================================================
    print("\n3. PUBMED API CALLS (Free - No Google quota used):")
    print("-" * 70)
    
    # Literature gathering for discussion
    pgs_searches = 5  # Top 5 PGS traits
    region_searches = 5  # Top 5 brain regions
    metric_searches = 4  # 4 network metrics
    general_searches = 4  # 4 general queries
    
    total_pubmed_calls = pgs_searches + region_searches + metric_searches + general_searches
    
    print(f"   - PGS-brain association searches: {pgs_searches}")
    print(f"   - Brain region function searches: {region_searches}")
    print(f"   - Network metric searches: {metric_searches}")
    print(f"   - General topic searches: {general_searches}")
    print(f"\n   TOTAL PUBMED API CALLS: {total_pubmed_calls}")
    print("   (These use NCBI E-utilities API, not Google API)")
    
    # ========================================================================
    # 4. Google API Quota Limits (gemini-flash-latest)
    # ========================================================================
    print("\n4. GOOGLE API QUOTA LIMITS (gemini-flash-latest):")
    print("-" * 70)
    print("\n   Free Tier Limits:")
    print("   - Requests per minute (RPM): 15")
    print("   - Requests per day (RPD): 1,500")
    print("   - Tokens per minute (TPM): 1,000,000")
    print("   - Tokens per day (TPD): No limit for free tier")
    
    # ========================================================================
    # 5. Quota Consumption Analysis
    # ========================================================================
    print("\n5. QUOTA CONSUMPTION PER RUN:")
    print("-" * 70)
    
    rpm_limit = 15
    rpd_limit = 1500
    tpm_limit = 1000000
    
    rpm_used_percent = (total_llm_calls / rpm_limit) * 100
    rpd_used_percent = (total_llm_calls / rpd_limit) * 100
    tpm_used_percent = (total_tokens / tpm_limit) * 100
    
    print(f"\n   Requests:")
    print(f"   - Used: {total_llm_calls} calls")
    print(f"   - RPM quota: {rpm_used_percent:.2f}% of {rpm_limit}")
    print(f"   - RPD quota: {rpd_used_percent:.2f}% of {rpd_limit}")
    
    print(f"\n   Tokens:")
    print(f"   - Used: ~{total_tokens:,} tokens")
    print(f"   - TPM quota: {tpm_used_percent:.2f}% of {tpm_limit:,}")
    
    # ========================================================================
    # 6. Cost Estimation (if using paid tier)
    # ========================================================================
    print("\n6. COST ESTIMATION (Pay-as-you-go pricing):")
    print("-" * 70)
    print("\n   gemini-flash-latest pricing:")
    print("   - Input: $0.075 per 1M tokens")
    print("   - Output: $0.30 per 1M tokens")
    
    input_tokens = results_input_tokens + discussion_input_tokens
    output_tokens = results_output_tokens + discussion_output_tokens
    
    input_cost = (input_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost
    
    print(f"\n   Per run cost:")
    print(f"   - Input: ${input_cost:.6f} ({input_tokens:,} tokens)")
    print(f"   - Output: ${output_cost:.6f} ({output_tokens:,} tokens)")
    print(f"   - TOTAL: ${total_cost:.6f} per run")
    
    # ========================================================================
    # 7. Batch Run Estimates
    # ========================================================================
    print("\n7. BATCH RUN ESTIMATES:")
    print("-" * 70)
    
    runs_per_day_rpm = int(60 / (total_llm_calls / rpm_limit))  # Based on RPM
    runs_per_day_rpd = int(rpd_limit / total_llm_calls)  # Based on RPD
    runs_per_day = min(runs_per_day_rpm, runs_per_day_rpd)
    
    print(f"\n   Maximum runs per day (free tier):")
    print(f"   - Limited by RPM: ~{runs_per_day_rpm} runs/day")
    print(f"   - Limited by RPD: ~{runs_per_day_rpd} runs/day")
    print(f"   - Practical limit: ~{runs_per_day} runs/day")
    print(f"   - Cost if paid tier: ${total_cost * runs_per_day:.2f}/day")
    
    # ========================================================================
    # 8. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPer single run:")
    print(f"  • Google Gemini API calls: {total_llm_calls}")
    print(f"  • Total tokens: ~{total_tokens:,}")
    print(f"  • RPM quota used: {rpm_used_percent:.1f}%")
    print(f"  • TPM quota used: {tpm_used_percent:.2f}%")
    print(f"  • Estimated cost (paid tier): ${total_cost:.6f}")
    print(f"\nFree tier allows: ~{runs_per_day} runs per day")
    print(f"PubMed API calls: {total_pubmed_calls} (free, unlimited)")
    print("\nNote: Actual usage may vary based on:")
    print("  - Actual output length generated by the model")
    print("  - Amount of literature retrieved from PubMed")
    print("  - Size of input CCA results")
    print("=" * 70)

if __name__ == "__main__":
    estimate_api_usage()
