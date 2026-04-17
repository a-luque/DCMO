import carla

from scenic.core.sensors import CallbackSensor

class CarlaCollisionSensor(CallbackSensor):
    def __init__(
        self,
        offset=(0, 0, 0),
        rotation=(0, 0, 0),
        attributes=None,
    ):
        super().__init__(defaultValue=0)
        self.offset = offset
        self.rotation = rotation

        if isinstance(attributes, str):
            raise NotImplementedError(
                "String parsing for attributes is not yet implemented. Feel free to do so."
            )
        elif isinstance(attributes, dict):
            self.attributes = attributes
        else:
            self.attributes = {}

        self.convert = None
        convert = self.attributes.get("convert")
        if convert is not None and not isinstance(convert, (str, int)):
            raise TypeError("'convert' has to be int or string.")
        self.convert = convert

        self.frame = 0

        self.blueprint = "sensor.other.collision"

        def onData(self, data):
            super().onData(data)

        


    def process(self, data):
        return data
