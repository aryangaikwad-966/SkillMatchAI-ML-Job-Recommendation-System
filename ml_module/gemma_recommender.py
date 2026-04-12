"""
Gemma 3 AI-Enhanced Recommender Module
=======================================

Uses Google's Gemma 3 model (via Ollama/Docker) to provide:
- AI-powered re-ranking of TF-IDF recommendations
- Natural language explanations for why each job matches
- Skill gap analysis and career insights

Requires: Docker container running Ollama with gemma3 model
Fallback: Gracefully degrades if Ollama is not available

Author: ML Engineer
Date: 2026-04-13
"""

import requests
import json
import time
from typing import List, Dict, Optional


OLLAMA_BASE_URL = "http://localhost:11434"


class GemmaRecommender:
    """
    Enhances job recommendations using Gemma 3 LLM.
    
    Takes TF-IDF based recommendations and uses Gemma 3 to:
    1. Generate natural language explanations
    2. Provide skill gap analysis
    3. Offer career advice based on matches
    """
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = "gemma3"):
        self.base_url = base_url
        self.model = model
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama/Gemma 3 is running and accessible."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                self._available = any(self.model in m for m in models)
                if not self._available:
                    print(f"⚠️  Ollama running but '{self.model}' not found.")
                    print(f"   Available models: {models}")
                    print(f"   Run: docker exec ollama-gemma3 ollama pull {self.model}")
                return self._available
        except (requests.ConnectionError, requests.Timeout):
            pass
        
        self._available = False
        print("⚠️  Ollama not reachable at", self.base_url)
        print("   Start with: docker compose up -d")
        return False
    
    def _chat(self, prompt: str, temperature: float = 0.3) -> str:
        """Send a prompt to Gemma 3 and return the response."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"   Error communicating with Gemma 3: {e}")
        return ""
    
    def enhance_recommendations(self, recommendations: List[Dict], 
                                 user_input: str) -> List[Dict]:
        """
        Enhance TF-IDF recommendations with AI-generated insights.
        
        Args:
            recommendations: List of recommendation dicts from TF-IDF system
            user_input: Original user skills input
            
        Returns:
            Enhanced recommendations with 'ai_explanation' field added
        """
        if not self.is_available():
            print("⏭️  Skipping AI enhancement (Gemma 3 not available)")
            for rec in recommendations:
                rec['ai_explanation'] = "N/A (Gemma 3 not running)"
            return recommendations
        
        print(f"\n🤖 Enhancing recommendations with Gemma 3...")
        start_time = time.time()
        
        for i, rec in enumerate(recommendations):
            print(f"   Processing recommendation {i+1}/{len(recommendations)}...", end=" ")
            
            prompt = self._build_prompt(rec, user_input)
            explanation = self._chat(prompt)
            rec['ai_explanation'] = explanation if explanation else "Could not generate explanation"
            
            print("✓")
        
        elapsed = time.time() - start_time
        print(f"\n✓ AI enhancement complete in {elapsed:.1f}s")
        
        return recommendations
    
    def _build_prompt(self, rec: Dict, user_input: str) -> str:
        """Build a focused prompt for Gemma 3."""
        return f"""You are a career advisor. A job seeker has these skills: {user_input}

They were matched to this job:
- Title: {rec['job_title']}
- Company: {rec.get('company', 'N/A')}
- Industry: {rec.get('industry', 'N/A')}
- Required Skills: {', '.join(rec.get('matched_skills', []))}
- Match Score: {rec['similarity_score']:.1%}

In 2-3 sentences, explain:
1. Why this job is a good match for their skills
2. What skills they might need to develop

Be concise and specific."""
    
    def get_career_summary(self, recommendations: List[Dict], 
                           user_input: str) -> str:
        """Generate an overall career analysis summary."""
        if not self.is_available():
            return "Career summary not available (Gemma 3 not running)"
        
        print("🤖 Generating career summary with Gemma 3...", end=" ")
        
        job_list = "\n".join([
            f"  {i+1}. {r['job_title']} ({r['similarity_score']:.1%} match)"
            for i, r in enumerate(recommendations)
        ])
        
        prompt = f"""You are a career advisor. A job seeker has these skills: {user_input}

Based on these top job matches:
{job_list}

Provide a brief career summary (3-4 sentences):
- What career direction suits them best
- Key strengths based on their skills
- Top 2 skills they should learn next

Be concise and actionable."""
        
        summary = self._chat(prompt, temperature=0.4)
        print("✓")
        return summary if summary else "Could not generate career summary"
    
    def status(self) -> Dict:
        """Get Ollama/Gemma 3 status information."""
        info = {
            "ollama_url": self.base_url,
            "model": self.model,
            "available": self.is_available()
        }
        
        if info["available"]:
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
                models = resp.json().get('models', [])
                for m in models:
                    if self.model in m.get('name', ''):
                        info["model_size"] = m.get('size', 'unknown')
                        info["model_family"] = m.get('details', {}).get('family', 'unknown')
                        break
            except:
                pass
        
        return info
