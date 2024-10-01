#set document(
  title: [ENCE464 T2 Poisson Assignment Group 13],
  author: "Jack Duignan",
  date: auto
)

#set page(
  paper: "a4",
  margin: (x: 1.8cm, y:1.5cm)
)

#set text(
  font: "Inria Serif",
  size: 12pt
)

#set par(
  justify: true,
  leading: 0.52em
)

#set table(stroke: (_, y) => if y == 0 { (bottom: 1pt, top: 1pt) })
#show table.cell.where(y: 0): set text(weight: "bold")

// use #show: appendix to being appendices
// #outline(target: heading.where(supplement: [Appendix]), title: [Appendix])
// 
#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#set math.equation(numbering: "(1)")

#show figure.where(
  kind: table
): set figure.caption(position: top)

#align(center, text(18pt)[
  ENCE464 Assignment 2: Computer Architecture
])

#align(center, text(14pt)[
  Group 13: Jack Duignan, Isaac Cone, Daniel Hawes
])

= Architecture Overview
The Central Processing Unit (CPU) described in this section is the AMD Ryzen 9 6900HX. Released in 2022, this CPU 8 identical cores with 2 threads per core for a total of 16 logical cores. This architecture is shown in @fig1 and will be described in further depth. 
#figure(image("./fig/cpu_topology.png", width: 60%), caption: "Central Processing Unit (CPU) architecture for the x86 AMD Ryzen 9 6900HX.")<fig1>

== Cores

== Memory Caching

= Multithreading

= Cache

= Profiling

= Compiler Optimisation

= Individual Topic 1 Jack Duignan

= Individual Topic 2 Isaac Cone

= Individual Topic 3 Daniel Hawes

#pagebreak()
#bibliography("bibliography.bib", title: "References", style: "institute-of-electrical-and-electronics-engineers")