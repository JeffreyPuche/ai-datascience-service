from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SexEnum(str, Enum):
    male = "male"
    female = "female"


class JobEnum(str, Enum):
    level_0 = "unskilled and non-resident"
    level_1 = "unskilled and resident"
    level_2 = "skilled"
    level_3 = "highly skilled"


class HousingEnum(str, Enum):
    own = "own"
    rent = "rent"
    free = "free"


class SavingAccountsEnum(str, Enum):
    na = "NA"
    little = "little"
    moderate = "moderate"
    quite_rich = "quite rich"
    rich = "rich"


class CheckingAccountEnum(str, Enum):
    na = "NA"
    little = "little"
    moderate = "moderate"
    rich = "rich"


class PurposeEnum(str, Enum):
    car = "car"
    furniture_equipment = "furniture/equipment"
    radio_tv = "radio/TV"
    domestic_appliances = "domestic appliances"
    repairs = "repairs"
    education = "education"
    business = "business"
    vacation_others = "vacation/others"


# input schemas
# Clase para el input del modelo de inferencia
class CreditRiskInput(BaseModel):
    """
    Define the structure of the input data for prediction.
    Field names must match the columns in the original dataset.
    """

    Age: int = Field(..., gt=0, description="Edad del solicitante en años.")
    Sex: SexEnum = Field(..., description="Sexo del solicitante.")
    Job: JobEnum = Field(..., description="Nivel de habilidad laboral.")
    Housing: HousingEnum = Field(..., description="Tipo de vivienda.")
    Saving_accounts: SavingAccountsEnum = Field(
        ..., alias="Saving accounts", description="Estado de la cuenta de ahorros."
    )
    Checking_account: CheckingAccountEnum = Field(
        ..., alias="Checking account", description="Estado de la cuenta corriente."
    )
    Credit_amount: float = Field(
        ..., gt=0, alias="Credit amount", description="Monto del crédito solicitado."
    )
    Duration: int = Field(..., gt=0, description="Duración del crédito en meses.")
    Purpose: PurposeEnum = Field(..., description="Propósito del crédito.")

    @field_validator("Job", mode="before")
    @classmethod
    def decode_job(cls, v):
        if isinstance(v, JobEnum):
            # Extract the number from the name (level_0 -> 0, level_1 -> 1, etc.)
            return int(v.name.split("_")[-1])
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "Sex": "male",
                "Job": "unskilled and resident",
                "Housing": "free",
                "Saving accounts": "NA",
                "Checking account": "NA",
                "Credit amount": 9055,
                "Duration": 36,
                "Purpose": "education",
            }
        },
    }


# output schemas
class CreditRiskOutput(BaseModel):
    """
    Define API response structure.
    """

    prediction: str = Field(..., description="Predicción del riesgo ('good' o 'bad').")
    probability: float = Field(
        ..., ge=0, le=1, description="Probabilidad de que el riesgo sea 'good'."
    )
