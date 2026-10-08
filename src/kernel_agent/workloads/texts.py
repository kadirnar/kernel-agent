"""Built-in texts of the LLM workload (inputs version 2, #170), written for kernel-agent.

* :data:`ESSAY`: the benchmark prompt, long expository prose (1,056 Qwen3 tokens); the
  prompt is its first ``prompt_len`` tokens, continued with :data:`STORY` when longer.
* :data:`STORY`: the held-out prompt, a narrative (614 tokens), continued with the essay.
* :data:`REQUESTS`: the diverse input set, requests of different kinds, languages and
  lengths (25 to 222 tokens) that end the prompt after a context cut from the essay.

No passage repeats inside a prompt of up to their combined length (about 1,670 tokens): a
greedy model that continues a prompt made of one paragraph repeated a few times repeats it
too, and prompt-lookup (n-gram) speculative decoding is then right almost every time
(40-67x on Qwen3-0.6B with the version-1 prompt, issue #170). A longer ``prompt_len``
repeats the text.
"""

from __future__ import annotations

ESSAY = """\
The history of computing is a story of abstraction layers. Engineers first wired logic \
gates by hand, then wrote machine code, then assembly, then compilers that translated high \
level languages into efficient instructions. Today, GPU kernels are written in domain \
specific languages that hide the hardware details while still exposing tiling, memory \
hierarchy and parallelism to the programmer.

The earliest electronic machines filled entire rooms and consumed as much power as a small \
factory. Programming them meant setting switches and plugging cables into patch panels, and \
a single calculation could take an afternoon to prepare. When stored programs arrived, \
instructions finally lived in the same memory as data, and a machine could be repurposed in \
minutes instead of days. That idea, simple as it sounds today, made software a craft of its \
own, separate from the design of the hardware.

Transistors replaced vacuum tubes in the late fifties, and integrated circuits soon packed \
dozens, then thousands, then millions of them onto a single sliver of silicon. Each \
generation was cheaper, faster and more reliable than the one before. Gordon Moore observed \
that the number of components on a chip doubled roughly every two years, and for half a \
century the industry treated his observation almost as a law of nature, planning road maps \
and factories around it.

For decades, programmers enjoyed a free lunch: the same program ran faster every year \
without any changes, simply because clock frequencies kept rising. Around the middle of the \
two thousands that lunch ended. Power density had reached the point where a faster clock \
would melt the chip, so manufacturers turned to parallelism instead. Processors gained more \
cores rather than more gigahertz, and software that could not split its work across them \
stopped getting faster.

Graphics processors had been parallel from the start. Rendering a frame means computing the \
colour of millions of pixels, and each pixel can be shaded independently of its neighbours. \
Early graphics cards implemented fixed pipelines in hardware, but by the early two thousands \
they exposed programmable shaders. Researchers noticed that a shader computing pixel colours \
could just as well simulate fluids or fold proteins, provided the problem was dressed up as \
a picture.

General purpose programming models removed the need for that disguise. A programmer could \
now launch a grid of thread blocks, each block sharing a small scratchpad of fast memory and \
each thread executing the same function on different data. The model was easy to learn and \
hard to master. Performance depended on details that sequential programmers had never needed \
to think about: whether neighbouring threads read neighbouring addresses, whether a block \
used too many registers, whether threads in the same warp took different branches.

Deep learning turned these details into an industry. Training a neural network is dominated \
by matrix multiplications and convolutions, operations with enormous arithmetic intensity \
that map naturally onto thousands of parallel arithmetic units. Vendors responded with \
specialised tensor cores that multiply small matrix tiles in a single instruction, and with \
reduced precision formats that trade a few bits of mantissa for twice or four times the \
throughput.

Yet raw arithmetic is rarely the whole story. A modern accelerator can perform far more \
operations per second than its memory can feed it, so many kernels spend most of their time \
waiting for data. The roofline model captures this trade-off in a single picture: a kernel \
is limited either by peak compute or by memory bandwidth, depending on how many operations \
it performs per byte it loads. Moving a kernel toward the roof usually means reusing data \
that has already been brought on chip, which is exactly what tiling achieves.

Inference adds its own twist. When a language model generates text one token at a time, \
each step multiplies a single vector by every weight matrix in the network. The arithmetic \
is trivial, but every weight must still travel from memory to the processor, so the step is \
bound by bandwidth and by the overhead of launching hundreds of small kernels. Kernel \
fusion, graph capture and persistent kernels attack that overhead directly, while \
quantisation shrinks the bytes that have to move.

Compilers have tried to automate all of this. Tensor compilers search over tilings, loop \
orders and fusion decisions, generating code that sometimes rivals hand written kernels. \
Their weakness is the long tail: unusual shapes, new hardware features and operations that \
fall outside the patterns the compiler knows. In practice, performance engineers combine \
both worlds, letting a compiler handle the common cases and writing custom kernels for the \
hot spots that matter most.

Measuring performance honestly is harder than it looks. A benchmark that runs the same \
input over and over can reward tricks that help nobody in production, such as caching \
results or specialising code paths for one particular input. Careful engineers therefore \
time their kernels on realistic data, check that outputs still match a trusted reference, \
and report the variation between runs rather than a single flattering number. Clock \
frequencies, temperature and other processes sharing the device all leave fingerprints in \
the measurements.

Looking ahead, the boundary between hardware and software keeps shifting. Chips gain \
asynchronous copy engines, hardware schedulers and new number formats, and each feature \
arrives first as a low level instruction that only experts can use. Over time, libraries \
and languages absorb it, and what once required a hand tuned kernel becomes a single \
function call. The cycle then repeats with the next generation, which is why the craft of \
writing fast kernels has never quite gone out of fashion."""

STORY = """\
A lighthouse keeper on a remote island kept a careful log of every storm, every passing \
ship and every change in the colour of the sea. Years later, sailors read the notebooks to \
learn which currents were safe and which harbours offered shelter when the weather turned \
without warning.

Her name was Ingrid, and she had come to the island at the age of twenty-three with two \
suitcases, a crate of books and a cat that refused to leave the boat for the first hour. \
The previous keeper had left a short note on the kitchen table: the lamp needs oil every \
evening, the radio works only when the wind blows from the south, and the goats belong to \
nobody.

The first winter tested her. Gales arrived from the north-west and stayed for weeks, \
throwing spray over the gallery rail and rattling the windows of the lantern room. Supplies \
came once a month when the sea allowed it, and twice that winter it did not. She learned to \
bake bread with half the flour, to mend the generator with wire from a fence, and to read \
by the glow of the great lens when the house lights failed.

In spring the island changed completely. Puffins returned to the cliffs, the grass turned a \
green so bright that it almost hurt the eyes, and fishing boats from the mainland anchored \
in the bay to wait for the tide. The fishermen brought newspapers, oranges and gossip, and \
in exchange Ingrid told them what she had seen: a whale off the southern reef, a drifting \
container, the strange calm that came every year before the first summer storm.

She began to draw maps in the margins of her log. One showed the sandbanks that moved after \
every winter, another the eddies behind the headland where small boats could hide from the \
swell. She noted the times of the tides beside the phases of the moon and soon found \
patterns that the official charts had missed. When a trawler ran aground in fog one autumn, \
its captain admitted that he had ignored her warning on the radio because it did not match \
his printed tables.

Visitors were rare, but they came. A geologist spent a summer measuring the rocks and left \
behind a hammer and a jar of fossils. A painter arrived for a week and stayed for three \
months, sketching the tower in every kind of light. Once, a boy who had run away from a \
boarding school on the mainland rowed across in a borrowed dinghy and asked whether he \
could become a keeper too. She fed him, gave him a job polishing brass, and called his \
parents on the radio when the wind finally turned south.

Automation reached the island when she was sixty. Engineers installed solar panels, a \
battery bank and a small computer that switched the lamp on at dusk and reported faults by \
satellite. They were polite and efficient and finished in four days. On the last evening, \
Ingrid climbed the stairs, watched the light come on by itself, and wrote the final entry \
in her log: lamp lit, wind light from the south-east, sea calm, nothing to report."""

#: The diverse input set of the LLM workload (``LLMWorkload.diverse_inputs``): label ->
#: a request that ends the prompt. Kinds, languages and lengths differ, and so does how
#: much a greedy continuation copies from its prompt (code and tables more than prose).
REQUESTS: dict[str, str] = {
    "news": (
        "CITY HALL, Tuesday. The municipal water authority announced on Tuesday that it "
        "will replace nearly forty kilometres of ageing pipes over the next three years, a "
        "project expected to cost 210 million euros. Officials said the work would begin in "
        "the northern districts, where leaks have caused repeated outages since last winter. "
        "Residents will receive a notice two weeks before the crews arrive on their street. "
        "Opposition councillors welcomed the plan but asked how it would be financed, noting "
        "that water tariffs already rose by six per cent in January. The director of the "
        "authority told reporters that"
    ),
    "dialogue": (
        "Customer: Hi, I ordered a pair of hiking boots last week, but the package still "
        "hasn't arrived.\n"
        "Agent: I'm sorry to hear that. Could you give me your order number, please?\n"
        "Customer: Sure, it's 58213.\n"
        "Agent: Thank you. I can see that the parcel left our warehouse on Monday and is "
        "now with the courier.\n"
        "Customer: Is there any way to find out when it will be delivered?\n"
        "Agent:"
    ),
    "code": (
        "def moving_average(values, window):\n"
        '    """Return the simple moving average of `values` over `window` items."""\n'
        "    if window <= 0:\n"
        '        raise ValueError("window must be positive")\n'
        "    result = []\n"
        "    total = 0.0\n"
        "    for i, value in enumerate(values):\n"
        "        total += value\n"
        "        if i >= window:\n"
        "            total -= values[i - window]\n"
        "        if i >= window - 1:\n"
        "            result.append(total / window)\n"
        "    return result\n"
        "\n"
        "\n"
        "def exponential_moving_average(values, alpha):\n"
        '    """Return the exponential moving average of `values` with smoothing `alpha`."""\n'
    ),
    "recipe": (
        "Lemon and herb roast chicken (serves four)\n\n"
        "Ingredients:\n"
        "- 1 whole chicken, about 1.6 kg\n"
        "- 2 lemons\n"
        "- 4 cloves of garlic\n"
        "- a small bunch of thyme\n"
        "- 3 tablespoons of olive oil\n"
        "- salt and black pepper\n\n"
        "Method:\n"
        "1. Heat the oven to 200 degrees and take the chicken out of the fridge thirty "
        "minutes before cooking.\n"
        "2. Zest one lemon and mix the zest with the oil, the crushed garlic and the chopped "
        "thyme.\n"
        "3."
    ),
    "poem": (
        "Autumn Ferry\n\n"
        "The ferry leaves at seven, grey on grey,\n"
        "the gulls already arguing the wake;\n"
        "a man with folded paper reads the day\n"
        "as if the headlines were a thing to take.\n\n"
        "Beyond the breakwater the water turns"
    ),
    "question": (
        "Question: Why does the sky look blue during the day but red and orange at sunset? "
        "Explain it simply.\nAnswer:"
    ),
    "math": (
        "Problem: A train leaves the station at 9:15 and travels at a constant speed of 84 "
        "kilometres per hour. A second train leaves the same station at 10:00 on a parallel "
        "track and travels at 105 kilometres per hour. At what time does the second train "
        "catch up with the first?\n\n"
        "Solution: In the 45 minutes before the second train departs, the first train covers"
    ),
    "table": (
        "date,city,temperature_c,humidity_pct,wind_kmh\n"
        "2024-03-01,Lisbon,16.2,71,14\n"
        "2024-03-01,Madrid,12.8,55,9\n"
        "2024-03-01,Paris,9.4,80,17\n"
        "2024-03-01,Berlin,6.1,76,21\n"
        "2024-03-02,Lisbon,17.0,68,11\n"
        "2024-03-02,Madrid,13.5,52,7\n"
        "2024-03-02,Paris,10.1,77,15\n"
        "2024-03-02,Berlin,5.7,81,24\n"
        "2024-03-03,Lisbon,"
    ),
    "german": (
        "Die kleine Bäckerei am Marktplatz öffnet jeden Morgen um sechs Uhr. Lange bevor die "
        "ersten Kunden kommen, steht der Bäcker schon in der warmen Backstube und knetet den "
        "Teig für Brot, Brötchen und Kuchen. Im Winter duftet die ganze Straße nach Zimt und "
        "Butter, und viele Leute machen auf dem Weg zur Arbeit einen kleinen Umweg, nur um "
        "an der offenen Tür vorbeizugehen. Seit über vierzig Jahren gehört die Bäckerei "
        "derselben Familie, und"
    ),
    "french": (
        "Le village se trouve au fond d'une vallée étroite, entouré de forêts de "
        "châtaigniers et de prairies en pente. Chaque samedi, le marché envahit la place de "
        "l'église : on y vend du fromage de chèvre, du miel de montagne, des légumes du "
        "jardin et parfois quelques outils anciens. Les habitants disent que le temps y "
        "passe plus lentement qu'ailleurs, et les visiteurs qui viennent pour un week-end "
        "finissent souvent par"
    ),
    "spanish": (
        "Durante el verano, la biblioteca municipal organiza talleres de lectura para niños "
        "y jóvenes. Cada semana, un autor invitado presenta un libro, responde preguntas y "
        "propone un pequeño ejercicio de escritura. Los participantes más pequeños dibujan a "
        "los personajes, mientras que los mayores escriben finales alternativos para las "
        "historias. Al terminar el programa, la biblioteca publica una antología con los "
        "mejores textos y"
    ),
    "chinese": (
        "春天来到小城的时候，河边的柳树先绿了。早上，老人们在公园里打太极拳，孩子们背着书包从"
        "桥上跑过。街角的小茶馆每天六点开门，老板总是先烧一壶开水，再把桌椅擦得干干净净。来喝"
        "茶的大多是附近的熟客，他们一边聊天，一边看着窗外的行人。有人说，这家茶馆已经开了五十"
        "多年，"
    ),
}
