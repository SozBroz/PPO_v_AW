# Ground truth: AWBW standard doctrine + cartridge AI rules

Collected 2026-06-10. Sources cited per section. Game-rule ground truth policy: AWBW wikis only; cartridge-AI internals from Wars World News (WWN) disassembly threads.

---

## 1. Ranked standard settings (AWBW)

Source: [Metagame — AWBW wiki](https://awbw.fandom.com/wiki/Metagame), [Global League](https://awbw.fandom.com/wiki/Global_League)

- 2 players, clear weather, **fog disabled**, CO powers on, tags off
- 0 starting funds, **1000g per property per turn**
- **Black Bomb banned**; Stealths frequently banned per-map
- **T0 (Broken Tier) COs always banned**: Hachi, Colin, Kanbei, Sensei, Grit; luck COs (Nell, Flak, Jugger) frequently banned
- CO tier randomly chosen at game creation; Map Committee can modify tiers/unit bans per map
- These match cart-game defaults except the bans

## 2. Game stages

Source: [Metagame — AWBW wiki](https://awbw.fandom.com/wiki/Metagame)

1. **Capture phase**: skill = capture efficiency. Infantry is the overwhelming build. **Neutral bases always captured first** (boosts unit production rate). On small maps infantry fight over contested properties early. Recon harass is situational outside fog.
2. **Midgame**: begins when tanks appear in numbers. Dominated by the **unit triangle — Tank, Anti-Air, B-Copter counter each other circularly** (Tank beats AA, AA beats B-Copter, B-Copter beats Tank). Vehicles interrupt undefended infantry captures. Players form shielded formations (expendable units in front), maneuver for unit count/value/composition/funding advantage, land decisive blows often with COP/SCOP.
3. **Lategame**: high-tech units (Md. Tank, Neotank, Bomber, Fighter) supplement foundations; remote properties get transports; pipe seams broken to open lanes.

## 3. Fundamentals (the competence checklist)

Source: [Basic Strategy Guide — AWBW wiki](https://awbw.fandom.com/wiki/Basic_Strategy_Guide)

Advantage categories: **Income, Army Value, Unit Count, Army Composition, Power Bar Charge, Positioning**.

- **Build from every factory every turn.** Unit count gates positioning; lots of infantry is correct (capture, shield, anti-infantry). Skipping an infantry to afford a key unit one turn earlier is acceptable midgame; airports/ports need not build every turn.
- **Efficient capture phase.** Take neutral bases ASAP; have early infantry capture multiple properties each; balance backline vs frontline capture order — don't arrive late to contested properties, don't capture frontline props you'll immediately lose.
- **First non-infantry unit is usually a Tank** (small/mid maps) — best at fighting for contested properties. Recon openings are FTA/STA-sensitive and fog-flavored; APC openings only when income acceleration offsets 5000g + defenselessness.
- **Spend your money; counter new units immediately.** Enemy copter → build AA (or own copter); first enemy tank → own tank; tech-up → respond. Banking is a lategame tool.
- **Power bar**: 1 star = 9000g lost or 18000g destroyed; stars inflate after each use; no charge while own power active. Overcharge enemy powers or deny constructive timing.
- **Positioning**: shield indirects, take heavy terrain, threaten counter-formations, fight for the center (reach multiple fronts), pick fronts to reinforce.

### Combat patterns (in escalating sophistication)

1. **Free attacks** — destroy more value than the counterattack returns (uncovered targets). Take them unless the unit is needed elsewhere or gets trapped.
2. **Attacks from better terrain** — when trading with covered units, terrain stars decide the multi-turn trade.
3. **Attacks while protecting the attacker** — e.g. tank+infantry 2HKO onto a city infantry, then re-occupy the city; copter attacks from inaccessible terrain blocking the AA's path; or kill the counter-units (all AAs) to win air superiority.
4. **Attacks with cover/backup** — attacker covered by your artillery/AA so the response loses more than it gains.
5. **Attacks with superior reinforcements** — trade wars won by closer production (forward airports) or central position.

Core skill above all: **damage-chart calculation** — OHKO/2HKO knowledge for own and opponent turns.

## 4. Meta unit composition (standard)

Source: [Metagame — AWBW wiki](https://awbw.fandom.com/wiki/Metagame)

- **Core**: Infantry (majority), Tank, B-Copter, Anti-Air
- **Support**: Artillery, Md. Tank
- **Rare/situational**: Bombers, Fighters, Neotanks (lategame, small fraction, never at the expense of unit count)
- **Transports** (APC, T-Copter, Black Boat, Lander): only for otherwise-unreachable properties or logistics; T-Copter/Black Boat most common
- **Naval combat units: rarely built.** Ground units are cheap, versatile, capture-relevant
- **Missiles: effectively absent from the meta.** Any RHEA build of Missiles/Cruisers on a map with no enemy air capability is a hard error

## 5. Cartridge AI (versus-mode bot) — decoded rules

Sources: [WWN "The AI analysis topic" (AWDS)](https://forums.warsworldnews.com/viewtopic.php?f=12&t=13928), [WWN "Core AW2 AI Behavior Hacking Thread"](https://forums.warsworldnews.com/viewtopic.php?p=417835) — ROM disassembly + empirical testing by ALAKTORN, Xenesis, DxDyDzD. This is the **floor of our competency bar** and the template for `rl/scripted_baseline.py`.

### Turn structure

- AI settings: Defense / General / Assault / Strike (War Room ≈ Assault). Decisions in up to **3 waves** over an internal unit list.
- **AW2 phase order** (Xenesis's disassembly): Turn-start COP → **Capture with Infantry** → Fire with Indirects → Attack with Fighters/Bombers → Attack with Directs → Attack with Infantry → Position with Transports → Position with Infantry Transports → Position with Indirects → Position with Landers → end-turn COP → Ends Turn. Once a unit commits in a phase it is skipped in later phases.

### Attack decision ("estimation")

- For each attackable target compute: **G-damage dealt − G-damage taken in counter**. Attack iff favorable or tie.
- **Unit value multipliers** (target valuation): Infantry ×2, Mech ×1.5, Mega Tank ×2, APC ×0.5 (per-AI-unit-type tables exist; these hold overall).
- **Limits**: if estimated damage ≤10% HP dealt, or counter ≥70% HP taken, only attack if G-trade is **4× favorable** (or tie). Exceptions: target about to finish HQ capture (always interrupt); 70% limit waived vs Mega Tank or in Strike behavior.
- Prefers attack position with most terrain stars (with exceptions); avoids clogging own factories; avoids tiles in range of enemy AA/indirects in some cases; low fuel → refuel instead.
- **Infantry exception**: ≥5HP infantry with a capturable property in range captures instead of estimating vs non-soldier directs (still attacks indirects/soldiers); 2-4HP infantry estimate everything before capturing; 1HP infantry retreat to heal.

### Build behavior (reactive)

- Builds **Mechs when you build Recons**; **Artillery when you have Tanks**; AA in response to air. Roughly 3/4 Infantry, 1/4 Mech when funds are infantry-tier.
- Builds counter-units at bases when enemy units are within (built unit's movement) tiles — even through impassable terrain (exploitable bug).
- Can be baited into bad builds (e.g. coastal Cruisers by a Black Boat near its port).

### Known weaknesses (what "without the embarrassing bits" means)

- **1-turn lookahead, purely reactive** — no plan; any human looking 2+ turns ahead beats it
- Predictable target obsession (infantry, APCs) — exploitable by human-shield baiting
- Often ignores terrain advantage in positioning
- Deterministic per-build "destiny" — same situation, same move

### What it still does right (and RHEA currently fumbles)

- Never idles factories
- Never knowingly takes a losing G-trade (estimation gate)
- Always captures with free infantry; always interrupts HQ captures
- Builds reactive counters to opponent composition

## 6. Local cached source dumps

Full page texts cached during research (machine-local, may be garbage-collected):

- Metagame wiki, FAQ, Basic Strategy Guide: fetched 2026-06-10, key content reproduced above
- WWN AI threads: estimation formula, phase list, value multipliers reproduced above; for deeper detail (per-unit value tables, AI vision hacking, RNG destiny mechanics) re-fetch the WWN URLs in section 5
