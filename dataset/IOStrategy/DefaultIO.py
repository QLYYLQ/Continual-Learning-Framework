from dataset.IOStrategy import ImageIOMeta, IORegistry


class BaseImage(metaclass=ImageIOMeta):  # type: ignore
    suffixes = ["jpg", "png", "jpeg"]

    def load(self):
        pass

    def write(self):
        pass


print(IORegistry)


class ImageTest(BaseImage):
    suffixes = ["...jpg"]

    def load(self):
        print("load")

    def write(self):
        print("write")


print(IORegistry)


class ImageTest2:
    suffixes = ["..png"]

    def load(self):
        pass

    def write(self):
        pass


BaseImage.register(ImageTest2)
print(IORegistry)
print(issubclass(ImageTest2, BaseImage))

if __name__ == "__main__":
    pass
