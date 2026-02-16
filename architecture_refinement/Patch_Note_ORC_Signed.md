====================================================================
Patch Note: Topological Entropy (TE) and Ollivier–Ricci Curvature (ORC)
Metric Definitions and Sign Handling
====================================================================

Context
-------
This patch clarifies and updates the mathematical definitions of the
training-free topology proxies used in NAS experiments, with the goal
of aligning precisely with prior work (Waqas et al., 2022) while
preserving the controlled setting of fixed-capacity graph search.

In particular, it corrects the handling of Ollivier–Ricci curvature
(ORC), whose sign carries semantic meaning related to robustness.

--------------------------------------------------------------------
Metric Definitions (Canonical)
--------------------------------------------------------------------

Let G = (V, E) be an undirected graph with |V| = N nodes.

--------------------------------------------------------------------
Topological Entropy (TE)
--------------------------------------------------------------------

Topological entropy is defined as the Shannon entropy of the empirical
degree distribution.

Let d_i be the degree of node i, and let:

    p(k) = |{ i ∈ V : d_i = k }| / N

Then the degree entropy is:

    H_deg(G) = − ∑_k p(k) log p(k)

Because graph size N is fixed in our experiments (N = 32), entropy is
normalized by a fixed constant to ensure numerical stability:

    TE(G) = H_deg(G) / C

where C is a fixed normalization constant (e.g., C = log N or an
empirically chosen upper bound). If clipping is applied, it is done
after normalization:

    TE(G) = clip( H_deg(G) / C , 0, 1 )

--------------------------------------------------------------------
Ollivier–Ricci Curvature (ORC)
--------------------------------------------------------------------

For each edge (u, v) ∈ E, the Ollivier–Ricci curvature is defined as:

    κ(u, v) = 1 − W_1(m_u, m_v) / d(u, v)

where:
- m_u and m_v are probability measures on the neighbors of u and v,
- W_1(·,·) is the Wasserstein-1 (Earth Mover’s) distance,
- d(u, v) is the graph distance between u and v.

Mean graph curvature is computed by averaging over edges:

    κ̄(G) = (1 / |E|) ∑_{(u,v) ∈ E} κ(u, v)

--------------------------------------------------------------------
Corrected ORC Proxy (Sign-Preserving)
--------------------------------------------------------------------

Previous implementations used the absolute value |κ̄(G)|, which
collapsed positive and negative curvature regimes.

This patch replaces that definition with a sign-preserving version:

    ORC(G) = κ̄(G)

This choice aligns with the interpretation in Waqas et al. (2022),
where more negative curvature corresponds to greater structural
fragility. 

--------------------------------------------------------------------
Rationale
--------------------------------------------------------------------

- Preserves meaningful geometric distinctions between graph topologies.
- Prevents collapse of ORC variance due to absolute-value clipping.
- Restores ORC’s influence in multi-objective NAS and candidate
  selection.
- Ensures mathematical and interpretive alignment with Waqas et al.
  (2022) while maintaining fixed-capacity experimental control.

--------------------------------------------------------------------
Scope
--------------------------------------------------------------------

- Affects training-free graph evaluation and NAS candidate selection.
- Does not alter graph generation, wiring, or model training code.
- Applicable to Plot 2 and subsequent NAS experiments.

--------------------------------------------------------------------
End of Patch Note
====================================================================
