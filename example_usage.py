"""
Simple GEMMA Provider Usage Examples for Teammates

Copy these patterns for your agents.
Remember: GEMMA doesn't follow instructions perfectly - always parse responses carefully!
"""

# ================================
# 1. BASIC GEMMA USAGE
# ================================

from gemma_provider import quick_invoke, gemma_provider

# Quick one-shot usage
response = quick_invoke(
    prompt="Kullanıcı talebi: Paket değiştirmek istiyorum",
    system_message="Sen müşteri taleplerini kategorize eden asistansın. PAKET, FATURA, TEKNIK veya DIGER yanıt ver.",
    temperature=0.1
)
print(response)

# Direct provider usage
response = gemma_provider.invoke_model(
    prompt="Kullanıcı: Faturamı ödeyeceğim",
    system_message="Sen Turkcell müşteri hizmetlerisindir.",
    temperature=0.3,
    max_tokens=512
)
print(response)

# ================================
# 2. SIMPLE AGENT TEMPLATE
# ================================

# from base_agent import BaseAgent

# class MyAgent(BaseAgent):
    
#     def __init__(self):
#         system_message = """
# Sen [ROLE] asistanısın. 
# [INSTRUCTIONS]
# Sadece [FORMAT] yanıt ver.
#         """.strip()
        
#         super().__init__(
#             agent_name="MyAgent",
#             system_message=system_message,
#             temperature=0.1,
#             max_tokens=256
#         )
    
#     def process(self, user_input: str) -> dict:
#         # Call GEMMA
#         raw_response = self._call_gemma(user_input)
        
#         # IMPORTANT: Parse response carefully (GEMMA doesn't follow instructions perfectly)
#         parsed_result = self._parse_response(raw_response)
        
#         return {
#             "status": "success",
#             "result": parsed_result
#         }
    
#     def _parse_response(self, response: str) -> str:
#         """Parse GEMMA response - customize this part!"""
#         response = response.strip().upper()
        
#         # Example: Classification parsing
#         if "SAFE" in response and "DANGER" not in response:
#             return "SAFE"
#         elif "DANGER" in response:
#             return "DANGER" 
#         else:
#             return "DANGER"  # Default safe choice

# # ================================
# # 3. USAGE
# # ================================

# # Create and use agent
# agent = MyAgent()
# result = agent.process("Test input")
# print(result)