# llm/prompts/base.py

class PromptContractError(Exception):
    pass


class PromptContract:

    READS = []
    WRITES = []
    SYSTEM = ""
    HUMAN_TEMPLATE = ""

    def validate_reads(self, state):
        missing = [k for k in self.READS if k not in state]
        if missing:
            raise PromptContractError(
                f"[PromptContract] Missing state fields for {self.__class__.__name__}: {missing}"
            )

    def validate_writes(self, output):
        if not isinstance(output, dict):
            raise PromptContractError(
                f"[PromptContract] {self.__class__.__name__} output must be dict."
            )

        missing = [k for k in self.WRITES if k not in output]
        if missing:
            raise PromptContractError(
                f"[PromptContract] Missing output fields for {self.__class__.__name__}: {missing}"
            )

    def build_system_prompt(self):
        return self.SYSTEM

    def build_user_prompt(self, state):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_user_prompt(state)"
        )

    def run(self, state):
        self.validate_reads(state)

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(state)

        return system_prompt, user_prompt
