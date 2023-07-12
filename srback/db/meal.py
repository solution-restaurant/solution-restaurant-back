from pydantic import BaseModel
class Meal(BaseModel):
    userRecoMeal: str
    userResOfAi: str
    userChkEat: str
    userMLD: str
    userAlarmImg: str