from datetime import timedelta

from fastapi import APIRouter

from models.user_model import UserModel
from services.user import create_user_serv
from src.models.auth_model import AuthModel
from src.services.auth_serv import get_auth_user_serv
from src.utils.bcrypt_handle import verified
from src.utils.error_handle import get_details_error
from src.utils.handle_respose import send_success_response
from src.utils.jwt_handle import generate_token

soil_moisture_router = APIRouter()


@soil_moisture_router.get("/evaluate_model")
def soil_moisture_evaluate_ctrl(data: AuthModel):
    try:
        user = get_auth_user_serv(data.email)

        if not user:
            return send_success_response(404, "Usuario no encontrado")

        return send_success_response(
            200,
            "Usuario login",
            {
                "token": "kjl",
                "user": {
                    "email": user.get("email"),
                    "name": user.get("name"),
                    "lastname": user.get("lastname"),
                    "phone": user.get("phone"),
                    "created_at": user.get("created_at"),
                },
            },
        )
    except Exception as error:
        return get_details_error(error)
