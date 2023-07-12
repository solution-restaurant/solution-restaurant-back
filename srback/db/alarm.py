from pydantic import BaseModel
class Alarm(BaseModel):
    userRecoMeal: str
    userResOfAi: str
    userChkEat: str
    userMLD: str
    userAlarmImg: str