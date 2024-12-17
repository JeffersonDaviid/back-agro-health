from src.database import User


def get_auth_user_serv(email: str):
    try:
        therapist = User.select().where(User.email == email)

        return therapist.dicts().first()
    except Exception as error:
        raise error
