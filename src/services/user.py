from models.user_model import UserModel
from src.database import User
from src.utils.bcrypt_handle import encrypt


def get_users_serv():
    try:
        users = User.select(
            User.email,
            User.name,
            User.lastname,
            User.phone,
            User.created_at,
        )
        return list(users.dicts())
    except Exception as error:
        raise error


def get_user_serv(email: str):
    try:
        user = User.select(
            User.email,
            User.name,
            User.lastname,
            User.phone,
            User.created_at,
        ).where(User.email == email)

        return user.dicts().first()
    except Exception as error:
        raise error


def create_user_serv(data: UserModel):
    try:
        passwordHashed = encrypt(data.password)

        user = User.create(
            email=data.email,
            name=data.name,
            lastname=data.lastname,
            password=passwordHashed,
            phone=data.phone,
        )
        return user
    except Exception as error:
        raise error
