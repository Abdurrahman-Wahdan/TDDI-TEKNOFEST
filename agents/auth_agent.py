# auth_agent.py
import os
import sys
import re
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from mcp.mcp_client import mcp_client

class AuthAgent(BaseAgent):
    def __init__(self):
        system_message = """
Sen Turkcell müşteri hizmetleri TC kimlik analiz asistanısın.

Kullanıcının cevabını analiz et ve şunlardan birini yanıtla:
- TC_PROVIDED: Kullanıcı TC kimlik numarası verdi (11 haneli sayı)
- TC_REFUSED: Kullanıcı TC kimlik vermeyi reddetti
- TC_CONFUSED: Kullanıcı TC kimlik neden gerekli olduğunu sorguluyor
- TC_PRIVACY: Kullanıcı gizlilik endişesi var
- TC_ALTERNATIVE: Kullanıcı başka yolla giriş yapmak istiyor
- TC_NOT_PROVIDED: Kullanıcı başka bir şey söyledi

SADECE yukarıdaki seçeneklerden birini döndür.
        """.strip()
        
        super().__init__(
            agent_name="AuthAgent",
            system_message=system_message,
            temperature=0.0,
            max_tokens=20
        )
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if context is None:
            context = {}
            
        try:
            # Get state from context
            auth_attempts = context.get("auth_attempts", 0)
            max_attempts = context.get("max_attempts", 1)
            
            # Call GEMMA to analyze user response
            gemma_response = self._call_gemma(user_input)
            response_type = self._parse_gemma_response(gemma_response)
            
            if response_type == "TC_PROVIDED":
                # Extract and verify TC kimlik
                tc_kimlik = self._extract_tc_kimlik(user_input)
                
                if not tc_kimlik:
                    return {
                        "status": "TC_INVALID",
                        "message": "TC kimlik formatı geçersiz. 11 haneli TC kimlik numaranızı doğru şekilde yazabilir misiniz?",
                        "state_updates": {
                            "awaiting_tc": True
                        }
                    }
                
                # Authenticate via MCP
                auth_result = mcp_client.authenticate_customer(tc_kimlik)
                
                if auth_result["success"] and auth_result["is_active"]:
                    return {
                        "status": "CUSTOMER",
                        "tc_kimlik": tc_kimlik,
                        "customer_id": auth_result["customer_id"],
                        "customer_data": auth_result["customer_data"],
                        "message": f"Hoş geldiniz {auth_result['customer_data']['first_name']} {auth_result['customer_data']['last_name']}!",
                        "state_updates": {
                            "is_authenticated": True,
                            "customer_id": auth_result["customer_id"],
                            "customer_data": auth_result["customer_data"],
                            "auth_attempts": 0,
                            "awaiting_tc": False
                        }
                    }
                elif auth_result["success"] and auth_result["exists"] and not auth_result["is_active"]:
                    return {
                        "status": "INACTIVE_CUSTOMER",
                        "tc_kimlik": tc_kimlik,
                        "customer_id": auth_result["customer_id"],
                        "customer_data": auth_result["customer_data"],
                        "message": "Hesabınız aktif değil. Müşteri hizmetlerimizi arayarak aktivasyon yapabilirsiniz: 532",
                        "state_updates": {
                            "auth_attempts": 0,
                            "awaiting_tc": False
                        }
                    }
                elif auth_result["success"] and not auth_result["exists"]:
                    return {
                        "status": "NOT_CUSTOMER",
                        "tc_kimlik": tc_kimlik,
                        "message": "Bu TC kimlik ile kayıtlı müşteri bulunamadı. Yeni müşteri olmak ister misiniz?",
                        "state_updates": {
                            "is_authenticated": False,
                            "is_non_customer": True,
                            "auth_attempts": 0,
                            "awaiting_tc": False
                        }
                    }
                else:
                    return {
                        "status": "ERROR",
                        "tc_kimlik": tc_kimlik,
                        "message": "Kimlik doğrulama sırasında hata oluştu. Lütfen tekrar dener misiniz?",
                        "state_updates": {
                            "awaiting_tc": True
                        }
                    }
            
            elif response_type == "TC_REFUSED":
                return self._handle_tc_refusal(auth_attempts, max_attempts)
            
            elif response_type == "TC_CONFUSED":
                return self._handle_tc_confusion()
            
            elif response_type == "TC_PRIVACY":
                return self._handle_privacy_concerns()
            
            elif response_type == "TC_ALTERNATIVE":
                return self._handle_alternative_request()
            
            else:
                return self._handle_unclear_response(auth_attempts, max_attempts)
                
        except Exception as e:
            return {
                "status": "ERROR",
                "message": "Sistem hatası oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }
    
    def _parse_gemma_response(self, gemma_response: str) -> str:
        """Parse GEMMA response with fallback logic"""
        response_upper = gemma_response.upper()
        
        if "TC_PROVIDED" in response_upper:
            return "TC_PROVIDED"
        elif "TC_REFUSED" in response_upper:
            return "TC_REFUSED"
        elif "TC_CONFUSED" in response_upper:
            return "TC_CONFUSED"
        elif "TC_PRIVACY" in response_upper:
            return "TC_PRIVACY"
        elif "TC_ALTERNATIVE" in response_upper:
            return "TC_ALTERNATIVE"
        else:
            return "TC_NOT_PROVIDED"
    
    def _handle_tc_refusal(self, auth_attempts: int, max_attempts: int) -> Dict[str, Any]:
        """Handle when user refuses to provide TC kimlik"""
        if auth_attempts == 0:
            return {
                "status": "REQUESTING_TC",
                "message": "Anlıyorum, gizlilik endişeniz olabilir. Ancak size kişisel hizmet verebilmem için TC kimlik numaranıza ihtiyacım var. Bu bilgi tamamen güvenli şekilde saklanır ve sadece kimlik doğrulama için kullanılır. Paylaşabilir misiniz?",
                "state_updates": {
                    "auth_attempts": auth_attempts + 1,
                    "awaiting_tc": True
                }
            }
        elif auth_attempts >= max_attempts:
            return {
                "status": "AUTH_FAILED",
                "message": "Anlıyorum, TC kimlik paylaşmak istemiyorsunuz. Bu durumda kişisel işlemlerinizi yapamam ama genel sorularınızı yanıtlayabilirim. Size nasıl yardımcı olabilirim?",
                "state_updates": {
                    "auth_failed": True,
                    "awaiting_tc": False,
                    "is_non_customer": True
                }
            }
        else:
            return {
                "status": "REQUESTING_TC",
                "message": "Tamam, anlıyorum. Ancak maalesef TC kimlik olmadan hesap bilgilerinize erişemem. Son bir kez sorayım - TC kimlik numaranızı paylaşır mısınız?",
                "state_updates": {
                    "auth_attempts": auth_attempts + 1,
                    "awaiting_tc": True
                }
            }
    
    def _handle_tc_confusion(self) -> Dict[str, Any]:
        """Handle when user is confused about why TC is needed"""
        return {
            "status": "REQUESTING_TC",
            "message": "TC kimlik numaranızı kimlik doğrulama için istiyorum. Bu sayede hesabınıza ait bilgileri güvenli şekilde size gösterebilirim - faturalarınızı, paketinizi, kullanımınızı. TC kimlik numaranızı paylaşabilir misiniz?",
            "state_updates": {
                "awaiting_tc": True
            }
        }
    
    def _handle_privacy_concerns(self) -> Dict[str, Any]:
        """Handle privacy concerns about TC kimlik"""
        return {
            "status": "REQUESTING_TC",
            "message": "Evet, bilgileriniz tamamen güvenli. TC kimlik numaranız sadece kimlik doğrulama için kullanılır ve şirket politikalarımız gereği hiçbir şekilde üçüncü kişilerle paylaşılmaz. Turkcell olarak müşteri bilgilerinin güvenliği bizim için çok önemli. TC kimlik numaranızı paylaşabilir misiniz?",
            "state_updates": {
                "awaiting_tc": True
            }
        }
    
    def _handle_alternative_request(self) -> Dict[str, Any]:
        """Handle requests for alternative authentication"""
        return {
            "status": "REQUESTING_TC",
            "message": "Maalesef şu anda sadece TC kimlik ile kimlik doğrulama yapabiliyorum. Telefon numarası veya diğer bilgilerle doğrulama sistemimizde henüz mevcut değil. TC kimlik numaranızı paylaşabilir misiniz?",
            "state_updates": {
                "awaiting_tc": True
            }
        }
    
    def _handle_unclear_response(self, auth_attempts: int, max_attempts: int) -> Dict[str, Any]:
        """Handle unclear responses"""
        if auth_attempts >= max_attempts:
            return {
                "status": "AUTH_FAILED",
                "message": "Anlaşılabilir bir yanıt alamadığım için devam edemiyorum. TC kimlik olmadan sadece genel sorularınızı yanıtlayabilirim. Nasıl yardımcı olabilirim?",
                "state_updates": {
                    "auth_failed": True,
                    "awaiting_tc": False,
                    "is_non_customer": True
                }
            }
        else:
            return {
                "status": "REQUESTING_TC",
                "message": "Size yardımcı olabilmem için 11 haneli TC kimlik numaranızı paylaşır mısınız? Örnek: 12345678901",
                "state_updates": {
                    "auth_attempts": auth_attempts + 1,
                    "awaiting_tc": True
                }
            }
    
    def _extract_tc_kimlik(self, user_input: str) -> Optional[str]:
        """Extract 11-digit Turkish ID from user input"""
        digits_only = re.sub(r'\D', '', user_input)
        
        if len(digits_only) >= 11:
            for i in range(len(digits_only) - 10):
                candidate = digits_only[i:i+11]
                if candidate[0] != '0':
                    return candidate
        return None

# Global instance
auth_agent = AuthAgent()

def authenticate_user(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    return auth_agent.process(user_input, context)

if __name__ == "__main__":
    # Example usage
    user_input = "Merhaba, TC kimliğimi vermek istemiyorum."
    response = authenticate_user(user_input)
    print(response)