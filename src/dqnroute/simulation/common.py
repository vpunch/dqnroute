import networkx as nx
import os

from typing import List
from simpy import Environment, Event, Interrupt
from ..event_series import EventSeries
from ..messages import *
from ..agents import *
from ..utils import *

class HandlerFactory:
    def __init__(self, **kwargs):
        super().__init__()
        self.centralized = False
        self.master_handler = None

    def _setEnv(self, env: DynamicEnv):
        self.env = env
        if self.centralized and (self.master_handler is None):
            self.master_handler = self.makeMasterHandler()
        self.ready()

    def _makeHandler(self, agent_id: AgentId) -> MessageHandler:
        if self.centralized:
            return SlaveHandler(id=agent_id, master=self.master_handler)
        else:
            neighbours = [v for _, v in self.env.conn_graph.edges(agent_id)]
            return self.makeHandler(agent_id, neighbours)

    def makeMasterHandler(self) -> MasterHandler:
        raise NotImplementedError()

    def makeHandler(self, agent_id: AgentId, neighbours: List[AgentId]) -> MessageHandler:
        raise NotImplementedError()

    def handlerClass(self, handler_type: str):
        raise NotImplementedError()

    def ready(self):
        pass

class MultiAgentEnv(ToDynEnv, HasLog):
    """
    Abstract class which simulates an environment with multiple agents,
    where agents are connected accordingly to a given connection graph.
    """
    def __init__(self, env: Environment, factory: HandlerFactory, **kwargs):
        self.env = env
        self.factory = factory
        self.conn_graph = self.makeConnGraph(**kwargs)

        agent_ids = list(self.conn_graph.nodes)

        self.factory._setEnv(self.toDynEnv())
        self.handlers = {agent_id: self.factory._makeHandler(agent_id) for agent_id in agent_ids}
        self.delayed_evs = {agent_id: {} for agent_id in agent_ids}

    def time(self):
        return self.env.now

    def logName(self):
        return 'World'

    def toDynEnv(self):
        return DynamicEnv(conn_graph=self.conn_graph, time=self.time)

    def makeConnGraph(self, **kwargs) -> nx.Graph:
        """
        A method which defines a connection graph for the system with
        given params.
        Should be overridden. The node labels of a resulting graph should be
        `AgentId`s.
        """
        raise NotImplementedError()

    def handle(self, from_agent: AgentId, event: WorldEvent) -> Event:
        """
        Main method which governs how events cause each other in the
        environment. Not to be overridden in children: `handleAction` and
        `handleWorldEvent` should be overridden instead.
        """
        if isinstance(event, Message):
            return self.handleMessage(from_agent, event)

        elif isinstance(event, Action):
            return self.handleAction(from_agent, event)

        elif isinstance(event, DelayedEvent):
            proc = self.env.process(self._delayedHandleGen(from_agent, event))
            self.delayed_evs[from_agent][event.id] = proc
            return Event(self.env).succeed()

        elif isinstance(event, DelayInterrupt):
            try:
                self.delayed_evs[from_agent][event.delay_id].interrupt()
            except (KeyError, RuntimeError):
                pass
            return Event(self.env).succeed()

        elif from_agent[0] == 'world':
            return handleWorldEvent(event)

        else:
            raise Exception('Non-world event: ' + str(event))

    def handleMessage(self, from_agent: AgentId, msg: Message) -> Event:
        """
        Method which handles how messages should be dealt with. Is not meant to be
        overridden.
        """
        if isinstance(msg, WireOutMsg):
            # Out message is considered to be handled as soon as its
            # handling by the recipient is scheduled. We do not
            # wait for other agent to handle them.
            self.env.process(self._handleOutMsgGen(from_agent, msg))
            return Event(self.env).succeed()
        else:
            raise UnsupportedMessageType(msg)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        """
        Method which governs how agents' actions influence the environment
        Should be overridden by subclasses.
        """
        raise UnsupportedActionType(action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        """
        Method which governs how events from outside influence the environment.
        Should be overridden by subclasses.
        """
        if isinstance(event, LinkUpdateEvent):
            return self.handleConnGraphChange(event)
        else:
            raise UnsupportedEventType(event)

    def passToAgent(self, agent: AgentId, event: WorldEvent) -> Event:
        """
        Let an agent react on event and handle all events produced by agent as
        a consequence.
        """
        evs = []
        for new_event in self.handlers[agent].handle(event):
            evs.append(self.handle(agent, new_event))
        return self.env.all_of(evs)

    def handleConnGraphChange(self, event: LinkUpdateEvent) -> Event:
        """
        Adds or removes the connection link and notifies the agents that
        the corresponding intefaces changed availability.
        Connection graph itself does not change to preserve interfaces numbering.
        """
        u = event.u
        v = event.v
        u_int = interface_idx(self.conn_graph, u, v)
        v_int = interface_idx(self.conn_graph, v, u)

        if isinstance(event, AddLinkEvent):
            u_ev = InterfaceSetupEvent(u_int, v, event.params)
            v_ev = InterfaceSetupEvent(v_int, u, event.params)
        elif isinstance(event, RemoveLinkEvent):
            u_ev = InterfaceShutdownEvent(u_int)
            v_ev = InterfaceShutdownEvent(v_int)
        return self.passToAgent(u, u_ev) & self.passToAgent(v, v_ev)

    def _delayedHandleGen(self, from_agent: AgentId, event: DelayedEvent):
        proc_id = event.id
        delay = event.delay
        inner = event.inner

        try:
            yield self.env.timeout(delay)
            self.handle(from_agent, inner)
        except Interrupt:
            pass
        del self.delayed_evs[from_agent][proc_id]

    def _handleOutMsgGen(self, from_agent: AgentId, msg: WireOutMsg):
        int_id = msg.interface
        inner = msg.payload
        to_agent, to_interface = resolve_interface(self.conn_graph, from_agent, int_id)
        yield self.passToAgent(to_agent, WireInMsg(to_interface, inner))


class SimulationRunner:
    """
    Class which constructs an environment from given settings and runs it.
    """

    def __init__(self, run_params, data_series: EventSeries, data_dir: str, **kwargs):
        self.run_params = run_params
        self.data_series = data_series
        self.data_dir = data_dir

        # Makes a world simulation
        self.env = Environment()
        self.factory = self.makeHandlerFactory(**kwargs)
        self.world = self.makeMultiAgentEnv(**kwargs)

    def runDataPath(self, random_seed) -> str:
        cfg = self.relevantConfig()
        return '{}/{}-{}.csv'.format(self.data_dir, data_digest(cfg), self.makeRunId(random_seed))

    def run(self, random_seed = None, ignore_saved = False,
            progress_step = None, progress_queue = None) -> EventSeries:
        """
        Runs the environment, optionally reporting the progress to a given queue
        """
        data_path = self.runDataPath(random_seed)
        run_id = self.makeRunId(random_seed)

        if not ignore_saved and os.path.isfile(data_path):
            self.data_series.load(data_path)
            if progress_queue is not None:
                progress_queue.put((run_id, None))

        else:
            self.env.process(self.runProcess(random_seed))

            if progress_queue is not None:
                if progress_step is None:
                    self.env.run()
                    progress_queue.put((run_id, progress_step))
                else:
                    next_step = progress_step
                    while self.env.peek() != float('inf'):
                        self.env.run(until=next_step)
                        progress_queue.put((run_id, progress_step))
                        next_step += progress_step
                    progress_queue.put((run_id, None))
            else:
                self.env.run()

            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.data_series.save(data_path)

        return self.data_series

    def makeHandlerFactory(self, **kwargs) -> MultiAgentEnv:
        """
        Makes a handler factory
        """
        raise NotImplementedError()

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        """
        Initializes a world environment.
        """
        raise NotImplementedError()

    def relevantConfig(self):
        """
        Defines a part of `run_params` which is used to calculate
        run hash (for data saving).
        """
        raise NotImplementedError()

    def makeRunId(self, random_seed):
        """
        Run identificator, which depends on random seed and some run params.
        """
        raise NotImplementedError()

    def runProcess(self, random_seed):
        """
        Generator which generates a series of test scenario events in
        the world environment.
        """
        raise NotImplementedError()
