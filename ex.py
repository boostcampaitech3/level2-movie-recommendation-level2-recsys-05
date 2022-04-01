from abc import ABC, abstractmethod


class Mother(ABC):
    @abstractmethod
    def show_capital(self):
        raise NotImplementedError("Should have implemented this")


class Me(Mother):
    def show_capital(self):
        print("너의 이름은?")

    def a_show_capital(self):
        print("나의 이름은?")


moder = Mother()
moder.show_capital()

me = Me()
me.show_capital()
