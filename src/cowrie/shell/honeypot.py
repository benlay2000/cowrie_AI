# Copyright (c) 2009-2014 Upi Tamminen <desaster@gmail.com>
# See the COPYRIGHT file for more information


from __future__ import annotations

import copy
import requests
import openai
import time
import os
import re
import shlex
from typing import Any

from twisted.internet import error
from twisted.python import failure, log
from twisted.python.compat import iterbytes

from cowrie.core.config import CowrieConfig
from cowrie.shell import fs
from cowrie.shell import protocol


class HoneyPotShell:
    def __init__(
        self, protocol: Any, interactive: bool = True, redirect: bool = False
    ) -> None:
        self.protocol = protocol
        self.interactive: bool = interactive
        self.redirect: bool = redirect  # to support output redirection
        self.cmdpending: list[list[str]] = []
        self.environ: dict[str, str] = copy.copy(protocol.environ)
        if hasattr(protocol.user, "windowSize"):
            self.environ["COLUMNS"] = str(protocol.user.windowSize[1])
            self.environ["LINES"] = str(protocol.user.windowSize[0])
        self.lexer: shlex.shlex | None = None
        self.showPrompt()
        self.invasion_start_time = None
        self.invasion_end_time = None

    def lineReceived(self, line: str) -> None:
        if self.invasion_start_time is None:
            # 记录入侵开始时间
            self.invasion_start_time = time.time()
        log.msg(eventid="cowrie.command.input", input=line, format="CMD: %(input)s")
        self.lexer = shlex.shlex(instream=line, punctuation_chars=True, posix=True)
        # Add these special characters that are not in the default lexer
        self.lexer.wordchars += "@%{}=$:+^,()`"

        tokens: list[str] = []

        while True:
            try:
                tokkie: str | None = self.lexer.get_token()
                # log.msg("tok: %s" % (repr(tok)))

                if tokkie is None:  # self.lexer.eof put None for mypy
                    if tokens:
                        self.cmdpending.append(tokens)
                    break
                else:
                    tok: str = tokkie

                # For now, treat && and || same as ;, just execute without checking return code
                if tok == "&&" or tok == "||":
                    if tokens:
                        self.cmdpending.append(tokens)
                        tokens = []
                        continue
                    else:
                        self.protocol.terminal.write(
                            f"-bash: syntax error near unexpected token `{tok}'\n".encode()
                        )
                        break
                elif tok == ";":
                    if tokens:
                        self.cmdpending.append(tokens)
                        tokens = []
                    continue
                elif tok == "$?":
                    tok = "0"
                elif tok[0] == "(":
                    cmd = self.do_command_substitution(tok)
                    tokens = cmd.split()
                    continue
                elif "$(" in tok or "`" in tok:
                    tok = self.do_command_substitution(tok)
                elif tok.startswith("${"):
                    envRex = re.compile(r"^\${([_a-zA-Z0-9]+)}$")
                    envSearch = envRex.search(tok)
                    if envSearch is not None:
                        envMatch = envSearch.group(1)
                        if envMatch in list(self.environ.keys()):
                            tok = self.environ[envMatch]
                        else:
                            continue
                elif tok.startswith("$"):
                    envRex = re.compile(r"^\$([_a-zA-Z0-9]+)$")
                    envSearch = envRex.search(tok)
                    if envSearch is not None:
                        envMatch = envSearch.group(1)
                        if envMatch in list(self.environ.keys()):
                            tok = self.environ[envMatch]
                        else:
                            continue

                tokens.append(tok)
            except Exception as e:
                self.protocol.terminal.write(
                    b"-bash: syntax error: unexpected end of file\n"
                )
                # Could run runCommand here, but i'll just clear the list instead
                log.msg(f"exception: {e}")
                self.cmdpending = []
                self.showPrompt()
                return

        if self.cmdpending:
            self.runCommand()
        else:
            self.showPrompt()

    def do_command_substitution(self, start_tok: str) -> str:
        """
        this performs command substitution, like replace $(ls) `ls`
        """
        result = ""
        if start_tok[0] == "(":
            # start parsing the (...) expression
            cmd_expr = start_tok
            pos = 1
        elif "$(" in start_tok:
            # split the first token to prefix and $(... part
            dollar_pos = start_tok.index("$(")
            result = start_tok[:dollar_pos]
            cmd_expr = start_tok[dollar_pos:]
            pos = 2
        elif "`" in start_tok:
            # split the first token to prefix and `... part
            backtick_pos = start_tok.index("`")
            result = start_tok[:backtick_pos]
            cmd_expr = start_tok[backtick_pos:]
            pos = 1
        else:
            log.msg(f"failed command substitution: {start_tok}")
            return start_tok

        opening_count = 1
        closing_count = 0

        # parse the remaining tokens and execute subshells
        while opening_count > closing_count:
            if cmd_expr[pos] in (")", "`"):
                # found an end of $(...) or `...`
                closing_count += 1
                if opening_count == closing_count:
                    if cmd_expr[0] == "(":
                        # return the command in () without executing it
                        result = cmd_expr[1:pos]
                    else:
                        # execute the command in $() or `` or () and return the output
                        result += self.run_subshell_command(cmd_expr[: pos + 1])

                    # check whether there are more command substitutions remaining
                    if pos < len(cmd_expr) - 1:
                        remainder = cmd_expr[pos + 1 :]
                        if "$(" in remainder or "`" in remainder:
                            result = self.do_command_substitution(result + remainder)
                        else:
                            result += remainder
                else:
                    pos += 1
            elif cmd_expr[pos : pos + 2] == "$(":
                # found a new $(...) expression
                opening_count += 1
                pos += 2
            else:
                if opening_count > closing_count and pos == len(cmd_expr) - 1:
                    if self.lexer:
                        tokkie = self.lexer.get_token()
                        if tokkie is None:  # self.lexer.eof put None for mypy
                            break
                        else:
                            cmd_expr = cmd_expr + " " + tokkie
                elif opening_count == closing_count:
                    result += cmd_expr[pos]
                pos += 1

        return result

    def run_subshell_command(self, cmd_expr: str) -> str:
        # extract the command from $(...) or `...` or (...) expression
        if cmd_expr.startswith("$("):
            cmd = cmd_expr[2:-1]
        else:
            cmd = cmd_expr[1:-1]

        # instantiate new shell with redirect output
        self.protocol.cmdstack.append(
            HoneyPotShell(self.protocol, interactive=False, redirect=True)
        )
        # call lineReceived method that indicates that we have some commands to parse
        self.protocol.cmdstack[-1].lineReceived(cmd)
        # remove the shell
        res = self.protocol.cmdstack.pop()
        try:
            output: str = res.protocol.pp.redirected_data.decode()[:-1]
        except AttributeError:
            return ""
        else:
            return output

    def runCommand(self):
        
        pp = None

        def runOrPrompt() -> None:
            if self.cmdpending:
                self.runCommand()
            else:
                self.showPrompt()

        def parse_arguments(arguments: list[str]) -> list[str]:
            parsed_arguments = []
            for arg in arguments:
                parsed_arguments.append(arg)

            return parsed_arguments

        def parse_file_arguments(arguments: str) -> list[str]:
            """
            Look up arguments in the file system
            """
            parsed_arguments = []
            for arg in arguments:
                matches = self.protocol.fs.resolve_path_wc(arg, self.protocol.cwd)
                if matches:
                    parsed_arguments.extend(matches)
                else:
                    parsed_arguments.append(arg)

            return parsed_arguments

        if not self.cmdpending:
            if self.protocol.pp.next_command is None:  # command dont have pipe(s)
                if self.interactive:
                    self.showPrompt()
                else:
                    # when commands passed to a shell via PIPE, we spawn a HoneyPotShell in none interactive mode
                    # if there are another shells on stack (cmdstack), let's just exit our new shell
                    # else close connection
                    if len(self.protocol.cmdstack) == 1:
                        ret = failure.Failure(error.ProcessDone(status=""))
                        self.protocol.terminal.transport.processEnded(ret)
                    else:
                        return
            else:
                pass  # command with pipes
            return

        cmdAndArgs = self.cmdpending.pop(0)
        cmd2 = copy.copy(cmdAndArgs)

        # Probably no reason to be this comprehensive for just PATH...
        environ = copy.copy(self.environ)
        cmd_array = []
        cmd: dict[str, Any] = {}
        while cmdAndArgs:
            piece = cmdAndArgs.pop(0)
            if piece.count("="):
                key, val = piece.split("=", 1)
                environ[key] = val
                continue
            cmd["command"] = piece
            cmd["rargs"] = []
            break

        if "command" not in cmd or not cmd["command"]:
            runOrPrompt()
            return

        pipe_indices = [i for i, x in enumerate(cmdAndArgs) if x == "|"]
        multipleCmdArgs: list[list[str]] = []
        pipe_indices.append(len(cmdAndArgs))
        start = 0

        # Gather all arguments with pipes

        for _index, pipe_indice in enumerate(pipe_indices):
            multipleCmdArgs.append(cmdAndArgs[start:pipe_indice])
            start = pipe_indice + 1

        cmd["rargs"] = parse_arguments(multipleCmdArgs.pop(0))
        # parse_file_arguments parses too much. should not parse every argument
        # cmd['rargs'] = parse_file_arguments(multipleCmdArgs.pop(0))
        cmd_array.append(cmd)
        cmd = {}

        for value in multipleCmdArgs:
            cmd["command"] = value.pop(0)
            cmd["rargs"] = parse_arguments(value)
            cmd_array.append(cmd)
            cmd = {}

        openai.api_key = "sk-HNqGUa4iK4uFD47F9cm7VnZeVJGTqNc7APPSvCBDuLBsD2A6"
        openai.api_base = "https://api.chatanywhere.tech/v1"  # 如果需要，设置你的基础 URL


        def gpt_35_api(messages: list):
            """为提供的对话消息创建新的回答。

            Args:
                messages (list): 完整的对话消息。
            """
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            print(completion.choices[0].message.content)

        def gpt_35_api_stream(messages: list):
            """为提供的对话消息创建新的回答 (流式传输)。

            Args:
                messages (list): 完整的对话消息。
            """
            stream = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                stream=True,
            )
            response_content = ""

            for chunk in stream:
                if chunk.choices[0].delta:  # 检查 delta 是否存在
                    content = chunk.choices[0].delta.get('content')  # 使用 get 方法安全访问
                    if content is not None:
                        print(content, end="")
                        response_content += content
            return response_content  # 返回完整的响应内容

        lastpp = None
        for index, cmd in reversed(list(enumerate(cmd_array))):
            cmdclass = self.protocol.getCommand(
                cmd["command"], environ["PATH"].split(":")
            )
            if cmdclass:
                log.msg(
                    input=cmd["command"] + " " + " ".join(cmd["rargs"]),
                    format="Command found: %(input)s",
                )
                if index == len(cmd_array) - 1:
                    lastpp = StdOutStdErrEmulationProtocol(
                        self.protocol, cmdclass, cmd["rargs"], None, None, self.redirect
                    )
                    pp = lastpp
                else:
                    pp = StdOutStdErrEmulationProtocol(
                        self.protocol,
                        cmdclass,
                        cmd["rargs"],
                        None,
                        lastpp,
                        self.redirect,
                    )
                    lastpp = pp
            else:
                log.msg(
                    eventid="cowrie.command.failed",
                    input=" ".join(cmd2),
                    format="Command not found: %(input)s",
                )
                input_command = "%(input)s"
                
                messages = [{'role': 'user', 'content': 'You are a virtual Linux system. Based on the input command "{}", guess what the output of a virtual Linux system might be.You do not need to actually execute the command.DONT explain anything.If you do not know the command just say "comand not found (the command)"'.format(cmd["command"])}]
                
                self.protocol.terminal.write(gpt_35_api_stream(messages).encode('utf-8') + b'\n')
                # self.protocol.terminal.write(
                #     "-bash: {}: command not found\n".format(cmd["command"]).encode("utf8")
                # )

                if (
                    isinstance(self.protocol, protocol.HoneyPotExecProtocol)
                    and not self.cmdpending
                ):
                    stat = failure.Failure(error.ProcessDone(status=""))
                    self.protocol.terminal.transport.processEnded(stat)

                runOrPrompt()
                pp = None  # Got a error. Don't run any piped commands
                break
        if pp:
            self.protocol.call_command(pp, cmdclass, *cmd_array[0]["rargs"])

    def resume(self) -> None:
        if self.interactive:
            self.protocol.setInsertMode()
        self.runCommand()

    def showPrompt(self) -> None:
        if not self.interactive:
            return

        prompt = ""
        if CowrieConfig.has_option("honeypot", "prompt"):
            prompt = CowrieConfig.get("honeypot", "prompt")
            prompt += " "
        else:
            cwd = self.protocol.cwd
            homelen = len(self.protocol.user.avatar.home)
            if cwd == self.protocol.user.avatar.home:
                cwd = "~"
            elif (
                len(cwd) > (homelen + 1)
                and cwd[: (homelen + 1)] == self.protocol.user.avatar.home + "/"
            ):
                cwd = "~" + cwd[homelen:]

            # Example: [root@svr03 ~]#   (More of a "CentOS" feel)
            # Example: root@svr03:~#     (More of a "Debian" feel)
            prompt = f"{self.protocol.user.username}@{self.protocol.hostname}:{cwd}"
            if not self.protocol.user.uid:
                prompt += "# "  # "Root" user
            else:
                prompt += "$ "  # "Non-Root" user

        self.protocol.terminal.write(prompt.encode("ascii"))
        self.protocol.ps = (prompt.encode("ascii"), b"> ")

    def eofReceived(self) -> None:
        """
        this should probably not go through ctrl-d, but use processprotocol to close stdin
        """
        log.msg("received eof, sending ctrl-d to command")
        if self.protocol.cmdstack:
            self.protocol.cmdstack[-1].handle_CTRL_D()

    def handle_CTRL_C(self) -> None:
        self.protocol.lineBuffer = []
        self.protocol.lineBufferIndex = 0
        self.protocol.terminal.write(b"\n")
        self.showPrompt()

    def handle_CTRL_D(self) -> None:
        log.msg("Received CTRL-D, exiting..")
        self.invasion_end_time = time.time()  # 记录入侵结束时间
        self.calculate_invasion_duration()
        stat = failure.Failure(error.ProcessDone(status=""))
        self.protocol.terminal.transport.processEnded(stat)
    
    def calculate_invasion_duration(self):
        if self.invasion_start_time and self.invasion_end_time:
            duration = self.invasion_end_time - self.invasion_start_time
            log.msg(f"Invasion duration: {duration:.2f} seconds")

    def handle_TAB(self) -> None:
        """
        lineBuffer is an array of bytes
        """
        if not self.protocol.lineBuffer:
            return

        line: bytes = b"".join(self.protocol.lineBuffer)
        if line[-1:] == b" ":
            clue = ""
        else:
            clue = line.split()[-1].decode("utf8")

        # clue now contains the string to complete or is empty.
        # line contains the buffer as bytes
        basedir = os.path.dirname(clue)
        if basedir and basedir[-1] != "/":
            basedir += "/"

        if not basedir:
            tmppath = self.protocol.cwd
        else:
            tmppath = basedir

        try:
            r = self.protocol.fs.resolve_path(tmppath, self.protocol.cwd)
        except Exception:
            return

        files = []
        for x in self.protocol.fs.get_path(r):
            if clue == "":
                files.append(x)
                continue
            if not x[fs.A_NAME].startswith(os.path.basename(clue)):
                continue
            files.append(x)

        if not files:
            return

        # Clear early so we can call showPrompt if needed
        for _i in range(self.protocol.lineBufferIndex):
            self.protocol.terminal.cursorBackward()
            self.protocol.terminal.deleteCharacter()

        newbuf = ""
        if len(files) == 1:
            newbuf = " ".join(
                line.decode("utf8").split()[:-1] + [f"{basedir}{files[0][fs.A_NAME]}"]
            )
            if files[0][fs.A_TYPE] == fs.T_DIR:
                newbuf += "/"
            else:
                newbuf += " "
            newbyt = newbuf.encode("utf8")
        else:
            if os.path.basename(clue):
                prefix = os.path.commonprefix([x[fs.A_NAME] for x in files])
            else:
                prefix = ""
            first = line.decode("utf8").split(" ")[:-1]
            newbuf = " ".join([*first, f"{basedir}{prefix}"])
            newbyt = newbuf.encode("utf8")
            if newbyt == b"".join(self.protocol.lineBuffer):
                self.protocol.terminal.write(b"\n")
                maxlen = max(len(x[fs.A_NAME]) for x in files) + 1
                perline = int(self.protocol.user.windowSize[1] / (maxlen + 1))
                count = 0
                for file in files:
                    if count == perline:
                        count = 0
                        self.protocol.terminal.write(b"\n")
                    self.protocol.terminal.write(
                        file[fs.A_NAME].ljust(maxlen).encode("utf8")
                    )
                    count += 1
                self.protocol.terminal.write(b"\n")
                self.showPrompt()

        self.protocol.lineBuffer = [y for x, y in enumerate(iterbytes(newbyt))]
        self.protocol.lineBufferIndex = len(self.protocol.lineBuffer)
        self.protocol.terminal.write(newbyt)


class StdOutStdErrEmulationProtocol:
    """
    Pipe support written by Dave Germiquet
    Support for commands chaining added by Ivan Korolev (@fe7ch)
    """

    __author__ = "davegermiquet"

    def __init__(
        self, protocol, cmd, cmdargs, input_data, next_command, redirect=False
    ):
        self.cmd = cmd
        self.cmdargs = cmdargs
        self.input_data: bytes = input_data
        self.next_command = next_command
        self.data: bytes = b""
        self.redirected_data: bytes = b""
        self.err_data: bytes = b""
        self.protocol = protocol
        self.redirect = redirect  # dont send to terminal if enabled

    def connectionMade(self) -> None:
        self.input_data = b""

    def outReceived(self, data: bytes) -> None:
        """
        Invoked when a command in the chain called 'write' method
        If we have a next command, pass the data via input_data field
        Else print data to the terminal
        """
        self.data = data

        if not self.next_command:
            if not self.redirect:
                if self.protocol is not None and self.protocol.terminal is not None:
                    self.protocol.terminal.write(data)
                else:
                    log.msg("Connection was probably lost. Could not write to terminal")
            else:
                self.redirected_data += self.data
        else:
            if self.next_command.input_data is None:
                self.next_command.input_data = self.data
            else:
                self.next_command.input_data += self.data

    def insert_command(self, command):
        """
        Insert the next command into the list.
        """
        command.next_command = self.next_command
        self.next_command = command

    def errReceived(self, data: bytes) -> None:
        if self.protocol and self.protocol.terminal:
            self.protocol.terminal.write(data)
        self.err_data = self.err_data + data

    def inConnectionLost(self) -> None:
        pass

    def outConnectionLost(self) -> None:
        """
        Called from HoneyPotBaseProtocol.call_command() to run a next command in the chain
        """

        if self.next_command:
            # self.next_command.input_data = self.data
            npcmd = self.next_command.cmd
            npcmdargs = self.next_command.cmdargs
            self.protocol.call_command(self.next_command, npcmd, *npcmdargs)

    def errConnectionLost(self) -> None:
        pass

    def processExited(self, reason: failure.Failure) -> None:
        log.msg(f"processExited for {self.cmd}, status {reason.value.exitCode}")

    def processEnded(self, reason: failure.Failure) -> None:
        log.msg(f"processEnded for {self.cmd}, status {reason.value.exitCode}")