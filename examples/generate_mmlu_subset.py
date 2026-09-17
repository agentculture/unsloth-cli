#!/usr/bin/env python3
"""Deterministic generator for ``eval/mmlu-subset.jsonl``, an MMLU-*style* suite.

**This is not MMLU.** No row here comes from the `cais/mmlu` or
`hails/mmlu_no_train` datasets, or from any other third-party corpus: every
question and every distractor below was written for this repository, and the
file carries the repository's own licence (MIT). It exists so the letter-choice
scoring path — and the shape of a multiple-choice benchmark — can be exercised
**offline, with no download**, on a machine that has never fetched MMLU. For the
real, quoted MMLU number, run ``sloth bench --benchmark mmlu``, which runs the
lm-evaluation-harness inside the NGC container.

Each row is a **task**-schema record::

    {"task": "mmlu-style multiple choice (astronomy)",
     "input": "<question>\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer with the letter.",
     "expected_output": "B"}

Because every ``expected_output`` is a single ``A``-``D`` letter, ``sloth eval``
switches the file into letter-choice scoring: the model's answer letter is
extracted with :func:`sloth.tune.scorers.extract_choice_letter` (tolerating
``"B"``, ``"B."``, ``"(B)"``, ``"Answer: B"``) and reported as ``choice_match``
per row and ``choice_acc_pct`` for the suite, alongside the unchanged
``exact_match`` numbers.

Everything here is pure stdlib and fully deterministic — no randomness, no clock
reads, no network — so running it twice produces byte-identical output, which
``tests/test_examples_suites.py`` asserts.

Usage::

    uv run python examples/generate_mmlu_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent

#: Path, relative to ``examples/``, of the generated suite.
RELATIVE_PATH = "eval/mmlu-subset.jsonl"

#: The instruction appended to every question, which is what makes a bare letter
#: the expected answer (and what ``extract_choice_letter`` reads back).
ANSWER_INSTRUCTION = "Answer with the letter."

#: The four option labels, in order.
LETTERS = ("A", "B", "C", "D")

#: ``{subject: [(question, (opt_a, opt_b, opt_c, opt_d), correct_index), ...]}``.
#: Ten original questions per subject, twelve subjects. Written for this repo:
#: general, non-proprietary knowledge, deliberately unambiguous so a competent
#: small model can be expected to score well above chance.
QUESTIONS: dict[str, list[tuple[str, tuple[str, str, str, str], int]]] = {
    "astronomy": [
        (
            "Which planet in our solar system has the most extensive ring system?",
            ("Jupiter", "Saturn", "Uranus", "Neptune"),
            1,
        ),
        (
            "What is the primary process that powers the Sun?",
            ("Chemical burning", "Nuclear fission", "Nuclear fusion", "Gravitational collapse"),
            2,
        ),
        ("A light-year is a unit of what?", ("Time", "Distance", "Brightness", "Mass"), 1),
        (
            "Which object is left behind by a supernova of a very massive star?",
            (
                "A white dwarf",
                "A brown dwarf",
                "A black hole or neutron star",
                "A planetary nebula",
            ),
            2,
        ),
        (
            "What causes the phases of the Moon?",
            (
                "Earth's shadow on the Moon",
                "The Moon's changing distance from Earth",
                "Clouds in Earth's atmosphere",
                "The changing angle between Sun, Moon and Earth",
            ),
            3,
        ),
        (
            "Which galaxy is the nearest large spiral galaxy to the Milky Way?",
            ("Andromeda", "Triangulum", "Sombrero", "Whirlpool"),
            0,
        ),
        (
            "What is the approximate age of the universe according to current cosmology?",
            ("4.5 billion years", "13.8 billion years", "100 billion years", "500 million years"),
            1,
        ),
        (
            "Which layer of the Sun do we normally see in visible light?",
            ("The core", "The corona", "The photosphere", "The chromosphere"),
            2,
        ),
        (
            "What does a redshift in a distant galaxy's spectrum indicate?",
            (
                "It is moving away from us",
                "It is moving toward us",
                "It is unusually hot",
                "It contains no hydrogen",
            ),
            0,
        ),
        (
            "Which planet is known for a persistent storm called the Great Red Spot?",
            ("Mars", "Venus", "Jupiter", "Saturn"),
            2,
        ),
    ],
    "biology": [
        (
            "Which organelle is the main site of ATP production in eukaryotic cells?",
            ("Ribosome", "Mitochondrion", "Golgi apparatus", "Lysosome"),
            1,
        ),
        (
            "What molecule carries the genetic instructions in most living organisms?",
            ("DNA", "Cellulose", "Glycogen", "Keratin"),
            0,
        ),
        (
            "Photosynthesis in plants primarily takes place in which structure?",
            ("Mitochondria", "Chloroplasts", "Vacuoles", "Nucleus"),
            1,
        ),
        (
            "Which blood cells are chiefly responsible for carrying oxygen?",
            ("Platelets", "White blood cells", "Red blood cells", "Plasma cells"),
            2,
        ),
        (
            "What is the basic structural and functional unit of the kidney?",
            ("The alveolus", "The nephron", "The neuron", "The villus"),
            1,
        ),
        (
            "Enzymes are best described as what kind of molecule?",
            ("Lipids", "Proteins", "Nucleic acids", "Simple sugars"),
            1,
        ),
        (
            "In classical genetics, an organism with two identical alleles is called what?",
            ("Heterozygous", "Homozygous", "Haploid", "Polyploid"),
            1,
        ),
        (
            "Which process produces gametes with half the chromosome number?",
            ("Mitosis", "Meiosis", "Binary fission", "Budding"),
            1,
        ),
        ("Which kingdom do mushrooms belong to?", ("Plantae", "Animalia", "Fungi", "Protista"), 2),
        (
            "What gas do plants take in from the air for photosynthesis?",
            ("Oxygen", "Nitrogen", "Carbon dioxide", "Methane"),
            2,
        ),
    ],
    "chemistry": [
        ("What is the chemical symbol for potassium?", ("P", "Pt", "K", "Po"), 2),
        ("How many protons does a carbon atom have?", ("4", "6", "12", "14"), 1),
        (
            "A solution with a pH of 3 is best described as what?",
            ("Strongly basic", "Weakly basic", "Neutral", "Acidic"),
            3,
        ),
        (
            "Which bond involves the sharing of electron pairs between atoms?",
            ("Ionic bond", "Covalent bond", "Hydrogen bond", "Metallic bond"),
            1,
        ),
        (
            "What is the molar mass of water, to the nearest whole number?",
            ("12 g/mol", "16 g/mol", "18 g/mol", "22 g/mol"),
            2,
        ),
        (
            "In the periodic table, elements in the same group share what?",
            (
                "The same number of neutrons",
                "The same number of valence electrons",
                "The same atomic mass",
                "The same density",
            ),
            1,
        ),
        (
            "Which gas makes up roughly 78 percent of Earth's atmosphere?",
            ("Oxygen", "Nitrogen", "Argon", "Carbon dioxide"),
            1,
        ),
        (
            "What is oxidation, in terms of electrons?",
            ("Gain of electrons", "Loss of electrons", "Gain of protons", "Loss of neutrons"),
            1,
        ),
        (
            "Which of these is an example of a physical, not chemical, change?",
            ("Rusting iron", "Burning paper", "Melting ice", "Baking bread"),
            2,
        ),
        (
            "Avogadro's number is approximately what value?",
            ("6.02e23", "3.14e10", "9.81e5", "1.60e-19"),
            0,
        ),
    ],
    "physics": [
        (
            "Newton's second law is most commonly written as which equation?",
            ("E = mc^2", "F = ma", "PV = nRT", "V = IR"),
            1,
        ),
        ("What is the SI unit of force?", ("Joule", "Watt", "Newton", "Pascal"), 2),
        (
            "Which quantity is conserved in an isolated system with no external forces?",
            ("Temperature", "Momentum", "Volume", "Pressure"),
            1,
        ),
        (
            "Approximately how fast does light travel in a vacuum?",
            ("3e5 m/s", "3e8 m/s", "3e10 m/s", "3e3 m/s"),
            1,
        ),
        (
            "Which form of energy does a compressed spring store?",
            ("Kinetic energy", "Thermal energy", "Elastic potential energy", "Nuclear energy"),
            2,
        ),
        (
            "Doubling the speed of a moving object multiplies its kinetic energy by what factor?",
            ("2", "3", "4", "8"),
            2,
        ),
        (
            "What does Ohm's law relate?",
            (
                "Force, mass and acceleration",
                "Voltage, current and resistance",
                "Pressure, volume and temperature",
                "Energy, mass and the speed of light",
            ),
            1,
        ),
        (
            "Which type of wave requires a medium to travel through?",
            ("Radio waves", "X-rays", "Sound waves", "Visible light"),
            2,
        ),
        (
            "The second law of thermodynamics concerns which quantity?",
            ("Entropy", "Charge", "Wavelength", "Frequency"),
            0,
        ),
        (
            "What happens to the wavelength of a wave when its frequency increases at "
            "constant speed?",
            ("It increases", "It decreases", "It stays the same", "It becomes zero"),
            1,
        ),
    ],
    "world_history": [
        ("The Magna Carta was sealed in which century?", ("11th", "13th", "15th", "17th"), 1),
        (
            "Which empire built an extensive road network centred on Cusco?",
            ("Aztec", "Inca", "Maya", "Olmec"),
            1,
        ),
        (
            "The printing press with movable metal type in Europe is credited to whom?",
            ("Leonardo da Vinci", "Johannes Gutenberg", "Galileo Galilei", "William Caxton"),
            1,
        ),
        (
            "The Industrial Revolution began in which country?",
            ("France", "Germany", "Britain", "The United States"),
            2,
        ),
        (
            "The Silk Road primarily connected which two regions?",
            (
                "East Asia and the Mediterranean",
                "West Africa and Brazil",
                "Scandinavia and Iceland",
                "Australia and Indonesia",
            ),
            0,
        ),
        (
            "Which event marked the traditional end of the Western Roman Empire?",
            (
                "The founding of Constantinople",
                "The deposition of Romulus Augustulus in 476",
                "The Battle of Actium",
                "The eruption of Vesuvius",
            ),
            1,
        ),
        ("The French Revolution began in which year?", ("1776", "1789", "1804", "1848"), 1),
        (
            "Hieroglyphic writing is most associated with which ancient civilisation?",
            ("Egypt", "Persia", "Phoenicia", "Assyria"),
            0,
        ),
        (
            "The Meiji Restoration transformed which country?",
            ("China", "Korea", "Japan", "Thailand"),
            2,
        ),
        (
            "The Treaty of Westphalia in 1648 ended which conflict?",
            (
                "The Hundred Years' War",
                "The Thirty Years' War",
                "The Napoleonic Wars",
                "The War of the Roses",
            ),
            1,
        ),
    ],
    "geography": [
        (
            "Which river is the longest in South America?",
            ("Paraná", "Orinoco", "Amazon", "Magdalena"),
            2,
        ),
        (
            "Mount Kilimanjaro is located in which country?",
            ("Kenya", "Tanzania", "Uganda", "Ethiopia"),
            1,
        ),
        (
            "Which desert is the largest hot desert in the world?",
            ("Gobi", "Kalahari", "Sahara", "Atacama"),
            2,
        ),
        ("What is the capital of Canada?", ("Toronto", "Vancouver", "Montreal", "Ottawa"), 3),
        (
            "The Strait of Gibraltar separates which two continents?",
            ("Europe and Africa", "Asia and Africa", "Europe and Asia", "Asia and Australia"),
            0,
        ),
        (
            "Which of these countries is landlocked?",
            ("Portugal", "Bolivia", "Vietnam", "Norway"),
            1,
        ),
        (
            "The Great Barrier Reef lies off the coast of which country?",
            ("Indonesia", "Australia", "Fiji", "Philippines"),
            1,
        ),
        (
            "Which line of latitude is at zero degrees?",
            ("The Tropic of Cancer", "The Arctic Circle", "The equator", "The prime meridian"),
            2,
        ),
        (
            "Lake Baikal, the world's deepest lake, is in which country?",
            ("Mongolia", "Kazakhstan", "Russia", "China"),
            2,
        ),
        (
            "Which ocean lies between Africa and Australia?",
            ("Atlantic", "Indian", "Pacific", "Arctic"),
            1,
        ),
    ],
    "mathematics": [
        ("What is the value of 7 factorial?", ("720", "1040", "5040", "40320"), 2),
        ("The derivative of x^3 with respect to x is what?", ("x^2", "3x^2", "3x", "x^4/4"), 1),
        (
            "What is the sum of the interior angles of a triangle in Euclidean geometry?",
            ("90 degrees", "180 degrees", "270 degrees", "360 degrees"),
            1,
        ),
        ("Which of these numbers is prime?", ("51", "57", "61", "63"), 2),
        (
            "If a fair coin is flipped twice, what is the probability of two heads?",
            ("1/2", "1/3", "1/4", "1/8"),
            2,
        ),
        ("What is the slope of the line through (0, 1) and (2, 5)?", ("1", "2", "3", "4"), 1),
        (
            "The integral of 1/x with respect to x is what?",
            ("ln|x| + C", "x^-2 + C", "1/(2x^2) + C", "e^x + C"),
            0,
        ),
        ("What is the median of the list 3, 9, 4, 1, 7?", ("3", "4", "7", "9"), 1),
        ("Solve for x: 2x + 6 = 20.", ("5", "7", "10", "13"), 1),
        (
            "What is the area of a circle with radius 3, in terms of pi?",
            ("3 pi", "6 pi", "9 pi", "12 pi"),
            2,
        ),
    ],
    "computer_science": [
        (
            "What is the average-case time complexity of binary search on a sorted array?",
            ("O(1)", "O(log n)", "O(n)", "O(n log n)"),
            1,
        ),
        (
            "Which data structure serves elements in last-in, first-out order?",
            ("Queue", "Stack", "Heap", "Linked list"),
            1,
        ),
        (
            "In Big-O notation, what is the worst-case complexity of quicksort?",
            ("O(n)", "O(n log n)", "O(n^2)", "O(2^n)"),
            2,
        ),
        (
            "What does HTTP status code 404 indicate?",
            ("Server error", "Not found", "Unauthorized", "Moved permanently"),
            1,
        ),
        ("Which of these is a lossless compression format?", ("JPEG", "MP3", "PNG", "MPEG"), 2),
        ("What is the decimal value of the binary number 1011?", ("9", "11", "13", "15"), 1),
        (
            "In relational databases, what does a primary key guarantee?",
            (
                "Rows are sorted",
                "Each row is uniquely identified",
                "Columns are indexed",
                "Foreign keys are valid",
            ),
            1,
        ),
        (
            "Which protocol resolves domain names to IP addresses?",
            ("DHCP", "DNS", "FTP", "SMTP"),
            1,
        ),
        (
            "What does a compiler do?",
            (
                "Executes source code line by line",
                "Translates source code into another language, often machine code",
                "Stores program state between runs",
                "Manages network connections",
            ),
            1,
        ),
        (
            "A hash table gives what average-case lookup complexity?",
            ("O(1)", "O(log n)", "O(n)", "O(n^2)"),
            0,
        ),
    ],
    "economics": [
        (
            "What does GDP measure?",
            (
                "The total value of goods and services produced in an economy",
                "The total money held in banks",
                "The number of people employed",
                "Government debt",
            ),
            0,
        ),
        (
            "If demand rises while supply is unchanged, what usually happens to price?",
            ("It falls", "It rises", "It stays the same", "It becomes zero"),
            1,
        ),
        (
            "Inflation is best described as what?",
            (
                "A general rise in the price level",
                "A fall in unemployment",
                "An increase in exports",
                "A rise in interest rates",
            ),
            0,
        ),
        (
            "What is opportunity cost?",
            (
                "The money spent on a purchase",
                "The value of the next best alternative forgone",
                "The cost of producing one more unit",
                "The tax paid on a transaction",
            ),
            1,
        ),
        (
            "A market with a single seller and no close substitutes is called what?",
            ("Perfect competition", "Oligopoly", "Monopoly", "Monopsony"),
            2,
        ),
        (
            "Which institution typically sets a country's benchmark interest rate?",
            (
                "The central bank",
                "The stock exchange",
                "The treasury department of a commercial bank",
                "The chamber of commerce",
            ),
            0,
        ),
        (
            "What does a progressive tax system mean?",
            (
                "Everyone pays the same amount",
                "Higher incomes are taxed at higher rates",
                "Only companies are taxed",
                "Tax rates fall as income rises",
            ),
            1,
        ),
        (
            "Comparative advantage explains the gains from what?",
            ("Trade", "Taxation", "Regulation", "Inflation"),
            0,
        ),
        (
            "What is a recession commonly defined as?",
            (
                "Any fall in the stock market",
                "A sustained period of declining economic output",
                "A rise in the minimum wage",
                "A government budget deficit",
            ),
            1,
        ),
        (
            "Marginal cost is the cost of what?",
            (
                "All units produced",
                "Producing one additional unit",
                "Fixed assets only",
                "Borrowing money",
            ),
            1,
        ),
    ],
    "psychology": [
        (
            "Classical conditioning is most associated with which researcher?",
            ("Ivan Pavlov", "Jean Piaget", "Carl Rogers", "Noam Chomsky"),
            0,
        ),
        (
            "Which memory store holds information for a very brief sensory instant?",
            ("Long-term memory", "Working memory", "Sensory memory", "Procedural memory"),
            2,
        ),
        (
            "What does the term 'confirmation bias' describe?",
            (
                "Favouring information that supports existing beliefs",
                "Remembering the first item in a list best",
                "Copying the behaviour of a group",
                "Attributing success to luck",
            ),
            0,
        ),
        (
            "Piaget's theory describes stages of what?",
            ("Moral panic", "Cognitive development", "Sleep cycles", "Language decay"),
            1,
        ),
        (
            "In operant conditioning, what does positive reinforcement do?",
            (
                "Adds a pleasant consequence to increase a behaviour",
                "Removes a pleasant consequence",
                "Adds an unpleasant consequence",
                "Has no effect on behaviour",
            ),
            0,
        ),
        (
            "The bystander effect predicts what?",
            (
                "People help faster in large crowds",
                "People are less likely to help when others are present",
                "People never help strangers",
                "People help only relatives",
            ),
            1,
        ),
        (
            "Which part of the brain is most associated with forming new long-term memories?",
            ("Cerebellum", "Hippocampus", "Medulla", "Occipital lobe"),
            1,
        ),
        (
            "A double-blind study controls primarily for what?",
            (
                "Sampling cost",
                "Expectancy effects in participants and experimenters",
                "Statistical power",
                "Data storage errors",
            ),
            1,
        ),
        (
            "Maslow's hierarchy of needs places which category at its base?",
            ("Self-actualisation", "Esteem", "Physiological needs", "Belonging"),
            2,
        ),
        (
            "What does 'correlation does not imply causation' warn against?",
            (
                "Assuming one variable causes another because they move together",
                "Using large samples",
                "Reporting negative results",
                "Measuring two variables at once",
            ),
            0,
        ),
    ],
    "logic": [
        (
            "If all A are B, and all B are C, what follows?",
            ("All A are C", "All C are A", "No A are C", "Some B are not C"),
            0,
        ),
        (
            "Which is the contrapositive of 'If it rains, the ground is wet'?",
            (
                "If the ground is wet, it rains",
                "If the ground is not wet, it did not rain",
                "If it does not rain, the ground is not wet",
                "It rains and the ground is wet",
            ),
            1,
        ),
        (
            "What logical fallacy attacks the person rather than the argument?",
            ("Ad hominem", "Straw man", "False dilemma", "Slippery slope"),
            0,
        ),
        (
            "The statement 'P and not P' is what?",
            ("A tautology", "A contradiction", "A contingency", "A valid inference"),
            1,
        ),
        (
            "In propositional logic, what is 'P or Q' called?",
            ("A conjunction", "A disjunction", "A conditional", "A negation"),
            1,
        ),
        (
            "An argument is valid when what is true?",
            (
                "Its premises are true",
                "Its conclusion cannot be false if its premises are true",
                "Its conclusion is popular",
                "It has exactly two premises",
            ),
            1,
        ),
        (
            "Which fallacy presents only two options when more exist?",
            ("Circular reasoning", "False dilemma", "Appeal to authority", "Hasty generalisation"),
            1,
        ),
        (
            "What does modus ponens conclude from 'If P then Q' and 'P'?",
            ("Not Q", "Q", "Not P", "P or Q"),
            1,
        ),
        (
            "A sound argument is one that is what?",
            (
                "Valid with true premises",
                "Valid with any premises",
                "Persuasive to most readers",
                "Free of technical terms",
            ),
            0,
        ),
        (
            "What is the negation of 'All swans are white'?",
            (
                "No swans are white",
                "At least one swan is not white",
                "All swans are black",
                "Some swans are white",
            ),
            1,
        ),
    ],
    "nutrition": [
        (
            "Which macronutrient provides roughly 9 calories per gram?",
            ("Protein", "Carbohydrate", "Fat", "Fibre"),
            2,
        ),
        (
            "Vitamin C deficiency causes which condition?",
            ("Rickets", "Scurvy", "Beriberi", "Pellagra"),
            1,
        ),
        (
            "Which mineral is the main component of haemoglobin's oxygen-binding site?",
            ("Calcium", "Iron", "Zinc", "Potassium"),
            1,
        ),
        (
            "Dietary fibre is best described as what?",
            (
                "A type of fat",
                "Indigestible plant carbohydrate",
                "A complete protein",
                "A fat-soluble vitamin",
            ),
            1,
        ),
        (
            "Which vitamin does the skin synthesise from sunlight?",
            ("Vitamin A", "Vitamin B12", "Vitamin D", "Vitamin K"),
            2,
        ),
        (
            "Which food group is the richest common source of complete protein?",
            ("Leafy greens", "Eggs and meat", "Refined sugar", "Vegetable oils"),
            1,
        ),
        (
            "Excess sodium intake is most strongly linked to which health risk?",
            ("High blood pressure", "Night blindness", "Anaemia", "Osteoporosis"),
            0,
        ),
        (
            "What is the primary role of carbohydrates in the diet?",
            ("Providing energy", "Building enzymes", "Insulating nerves", "Transporting oxygen"),
            0,
        ),
        (
            "Calcium is most important for which body system?",
            ("Bones and teeth", "Lung capacity", "Eyesight", "Hair growth"),
            0,
        ),
        (
            "Which of these is a fat-soluble vitamin?",
            ("Vitamin C", "Vitamin B1", "Vitamin A", "Vitamin B6"),
            2,
        ),
    ],
}


def render_input(question: str, options: "tuple[str, ...]") -> str:
    """Render one question plus its lettered options into the suite's ``input``."""
    lines = [question]
    lines += [f"{letter}. {option}" for letter, option in zip(LETTERS, options)]
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def rotate_to(options: tuple[str, str, str, str], correct: int, target: int) -> tuple[str, ...]:
    """Cyclically rotate *options* so the item at *correct* lands at *target*.

    The four options are an unordered set, so rotating them changes nothing
    about the question — but it lets :func:`build_rows` spread the correct
    letter evenly across ``A``-``D``. Without this, the answer key written above
    happens to be ~50% ``B``, and a model that always says ``B`` would look
    competent. Deterministic: a pure index rotation, no shuffling.
    """
    shift = (target - correct) % len(LETTERS)
    return tuple(options[(index - shift) % len(LETTERS)] for index in range(len(LETTERS)))


def build_rows() -> list[dict[str, str]]:
    """Build every task-schema row, subjects in declaration order.

    The correct letter cycles ``A``, ``B``, ``C``, ``D`` across the whole file
    (see :func:`rotate_to`), so the answer key is exactly balanced and guessing a
    fixed letter scores chance.
    """
    rows: list[dict[str, str]] = []
    position = 0
    for subject, items in QUESTIONS.items():
        for question, options, correct in items:
            target = position % len(LETTERS)
            rotated = rotate_to(options, correct, target)
            rows.append(
                {
                    "task": f"mmlu-style multiple choice ({subject})",
                    "input": render_input(question, rotated),
                    "expected_output": LETTERS[target],
                }
            )
            position += 1
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """Write *rows* to *path* as JSONL, one compact object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def generate(target_dir: Path | None = None) -> dict[str, int]:
    """Write the suite under *target_dir* (default: ``examples/``).

    Returns a ``{relative path: row count}`` mapping, matching
    ``generate_suites.generate``'s contract.
    """
    base = Path(target_dir) if target_dir is not None else EXAMPLES_DIR
    rows = build_rows()
    write_jsonl(base / RELATIVE_PATH, rows)
    return {RELATIVE_PATH: len(rows)}


def main() -> int:
    """Regenerate the suite and print the row count."""
    for relative, count in generate().items():
        print(f"{relative}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
