# The algorithmic differentation tool pyadjoint and add-ons

The full documentation is available [here](http://pyadjoint.readthedocs.io)


# Installation
First install [FEniCS](http://fenicsproject.org) or [Firedrake](http://firedrakeproject.org)

Then install the pyadjoint with:
    pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@master

# Reporting bugs

If you found a bug, create an [issue].

[issue]: https://bitbucket.org/dolfin-adjoint/pyadjoint/issues/new

# Contributing

We love pull requests from everyone. 

Fork, then clone the repository:

    git clone https://bitbucket.org/dolfin-adjoint/pyadjoint.git

Make sure the tests pass:

    py.test tests

Make your change. Add tests for your change. Make the tests pass:

    py.test tests

Push to your fork and [submit a pull request][pr].

[pr]: https://bitbucket.org/dolfin-adjoint/pyadjoint/pull-requests/new

At this point you're waiting on us. We may suggest
some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted:

* Write tests.
* Add Python docstrings that follow the [Google Style][style].
* Write good commit and pull request message.

[style]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

# License
This software is licensed under the [GNU LGPL v3][license].

[license]: https://bitbucket.org/dolfin-adjoint/pyadjoint/raw/master/LICENSE