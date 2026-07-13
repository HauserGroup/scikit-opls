# Scikit-learn Developer’s Guide

Combined from the scikit-learn 1.9.0 Developer’s Guide (stable documentation), retrieved July 13, 2026. Original documentation © scikit-learn developers and distributed under the BSD 3-Clause License.

**Source index:** <https://scikit-learn.org/stable/developers/index.html>

## Contents

1. [Contributing](#contributing)
1. [Set up your development environment](#set-up-your-development-environment)
1. [Crafting a minimal reproducer for scikit-learn](#crafting-a-minimal-reproducer-for-scikit-learn)
1. [Developing scikit-learn estimators](#developing-scikit-learn-estimators)
1. [Developers’ Tips and Tricks](#developers-tips-and-tricks)
1. [Utilities for Developers](#utilities-for-developers)
1. [How to optimize for speed](#how-to-optimize-for-speed)
1. [Cython Best Practices, Conventions and Knowledge](#cython-best-practices-conventions-and-knowledge)
1. [Miscellaneous information / Troubleshooting](#miscellaneous-information--troubleshooting)
1. [Bug triaging and issue curation](#bug-triaging-and-issue-curation)
1. [Maintainer Information](#maintainer-information)
1. [Developing with the Plotting API](#developing-with-the-plotting-api)
1. [Developing with the callback API](#developing-with-the-callback-api)
1. [Implementing callback support in estimators](#implementing-callback-support-in-estimators)
1. [Developing callbacks](#developing-callbacks)

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/contributing.html>

## Contributing

This project is a community effort, shaped by a large number of contributors from across the world. For more information on the history and people behind scikit-learn see [About us](https://scikit-learn.org/stable/about.html#about). It is hosted on [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn). The decision making process and governance structure of scikit-learn is laid out in [Scikit-learn governance and decision-making](https://scikit-learn.org/stable/governance.html#governance).

Scikit-learn is [selective](https://scikit-learn.org/stable/faq.html#selectiveness) when it comes to adding new algorithms and features. This means the best way to contribute and help the project is to start working on known issues. See [Ways to contribute](https://scikit-learn.org/stable/developers/contributing.html#ways-to-contribute) to learn how to make meaningful contributions.

**Our community, our values**

We are a community based on openness and friendly, didactic discussions.

We aspire to treat everybody equally, and value their contributions. We are particularly seeking people from underrepresented backgrounds in Open Source Software and scikit-learn in particular to participate and contribute their expertise and experience.

Decisions are made based on technical merit and consensus.

Code is not the only way to help the project. Reviewing pull requests, answering questions to help others on mailing lists or issues, organizing and teaching tutorials, working on the website, improving the documentation, are all priceless contributions.

Communications on all channels should respect our [Code of Conduct](https://github.com/scikit-learn/scikit-learn/blob/main/CODE_OF_CONDUCT.md).

### Ways to contribute

There are many ways to contribute to scikit-learn. These include:

- referencing scikit-learn from your blog and articles, linking to it from your website, or simply [staring it](https://docs.github.com/en/get-started/exploring-projects-on-github/saving-repositories-with-stars) to say “I use it”; this helps us promote the project

- [improving and investigating issues](https://scikit-learn.org/stable/developers/bug_triaging.html#bug-triaging)

- [reviewing other developers’ pull requests](https://scikit-learn.org/stable/developers/contributing.html#code-review)

- reporting difficulties when using this package by submitting an [issue](https://github.com/scikit-learn/scikit-learn/issues), and giving a “thumbs up” on issues that others reported and that are relevant to you (see [Submitting a bug report or a feature request](https://scikit-learn.org/stable/developers/contributing.html#submitting-bug-feature) for details)

- improving the [Documentation](https://scikit-learn.org/stable/developers/contributing.html#contribute-documentation)

- making a code contribution

There are many ways to contribute without writing code, and we value these contributions just as highly as code contributions. If you are interested in making a code contribution, please keep in mind that scikit-learn has evolved into a mature and complex project since its inception in 2007. Contributing to the project code generally requires advanced skills, and it may not be the best place to begin if you are new to open source contribution. In this case we suggest you follow the suggestions in [New Contributors](https://scikit-learn.org/stable/developers/contributing.html#new-contributors).

Contributing to related projects

Scikit-learn thrives in an ecosystem of several related projects, which also may have relevant issues to work on, including smaller projects such as:

- [scikit-learn-contrib](https://github.com/search?q=org%3Ascikit-learn-contrib+is%3Aissue+is%3Aopen+sort%3Aupdated-desc&type=Issues)

- [joblib](https://github.com/joblib/joblib/issues)

- [sphinx-gallery](https://github.com/sphinx-gallery/sphinx-gallery/issues)

- [numpydoc](https://github.com/numpy/numpydoc/issues)

- [liac-arff](https://github.com/renatopp/liac-arff/issues)

and larger projects:

- [numpy](https://github.com/numpy/numpy/issues)

- [scipy](https://github.com/scipy/scipy/issues)

- [matplotlib](https://github.com/matplotlib/matplotlib/issues)

- and so on.

Look for issues marked “help wanted” or similar. Helping these projects may help scikit-learn too. See also [Related Projects](https://scikit-learn.org/stable/related_projects.html#related-projects).

#### New Contributors

We recommend new contributors start by reading this contributing guide, in particular [Ways to contribute](https://scikit-learn.org/stable/developers/contributing.html#ways-to-contribute), [Automated Contributions Policy](https://scikit-learn.org/stable/developers/contributing.html#automated-contributions-policy).

Next, we advise new contributors gain foundational knowledge on scikit-learn and open source by:

- [improving and investigating issues](https://scikit-learn.org/stable/developers/bug_triaging.html#bug-triaging)

  - confirming that a problem reported can be reproduced and providing a [minimal reproducible code](https://scikit-learn.org/stable/developers/minimal_reproducer.html#minimal-reproducer) (if missing), can help you learn about different use cases and user needs

  - investigating the root cause of an issue will aid you in familiarising yourself with the scikit-learn codebase

- [reviewing other developers’ pull requests](https://scikit-learn.org/stable/developers/contributing.html#code-review) will help you develop an understanding of the requirements and quality expected of contributions

- improving the [Documentation](https://scikit-learn.org/stable/developers/contributing.html#contribute-documentation) can help deepen your knowledge of the statistical concepts behind models and functions, and scikit-learn API

If you wish to make code contributions after building your foundational knowledge, we recommend you start by looking for an issue that is of interest to you, in an area you are already familiar with as a user or have background knowledge of. We recommend starting with smaller pull requests and following our [Pull request checklist](https://scikit-learn.org/stable/developers/contributing.html#pr-checklist). For expected etiquette around which issues and stalled PRs to work on, please read [Stalled pull requests](https://scikit-learn.org/stable/developers/contributing.html#stalled-pull-request), [Stalled and Unclaimed Issues](https://scikit-learn.org/stable/developers/contributing.html#stalled-unclaimed-issues) and [Issues tagged “Needs Triage”](https://scikit-learn.org/stable/developers/contributing.html#issues-tagged-needs-triage).

We rarely use the “good first issue” label because it is difficult to make assumptions about new contributors and these issues often prove more complex than originally anticipated. It is still useful to check if there are [“good first issues”](https://github.com/scikit-learn/scikit-learn/labels/good%20first%20issue), though note that these may still be time consuming to solve, depending on your prior experience.

For more experienced scikit-learn contributors, issues labeled [“Easy”](https://github.com/scikit-learn/scikit-learn/labels/Easy) may be a good place to look.

### Automated Contributions Policy

Contributing to scikit-learn requires human judgment, contextual understanding, and familiarity with scikit-learn’s structure and goals. It is not suitable for automatic processing by AI tools.

Please refrain from submitting issues or pull requests generated by fully-automated tools. Maintainers reserve the right, at their sole discretion, to close such submissions and to block any account responsible for them.

Review all code or documentation changes made by AI tools and make sure you understand all changes and can explain them on request, before submitting them under your name. Do not submit any AI-generated code that you haven’t personally reviewed, understood and tested, as this wastes maintainers’ time.

Please do not paste AI generated text in the description of issues, PRs or in comments as this makes it harder for reviewers to assess your contribution. We are happy for it to be used to improve grammar or if you are not a native English speaker.

If you used AI tools, please state so in your PR description.

PRs that appear to violate this policy will be closed without review.

### Submitting a bug report or a feature request

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a ticket to the [Bug Tracker](https://github.com/scikit-learn/scikit-learn/issues). You are also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/scikit-learn/scikit-learn/issues?q=) or [pull requests](https://github.com/scikit-learn/scikit-learn/pulls?q=).

- If you are submitting an algorithm or feature request, please verify that the algorithm fulfills our [new algorithm requirements](https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms).

- If you are submitting a bug report, we strongly encourage you to follow the guidelines in [How to make a good bug report](https://scikit-learn.org/stable/developers/contributing.html#filing-bugs).

When a feature request involves changes to the API principles or changes to dependencies or supported versions, it must be backed by a [SLEP](https://scikit-learn.org/stable/governance.html#slep), which must be submitted as a pull-request to [enhancement proposals](https://scikit-learn-enhancement-proposals.readthedocs.io) using the [SLEP template](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep_template.html) and follows the decision-making process outlined in [Scikit-learn governance and decision-making](https://scikit-learn.org/stable/governance.html#governance).

#### How to make a good bug report

When you submit an issue to [GitHub](https://github.com/scikit-learn/scikit-learn/issues), please do your best to follow these guidelines! This will make it a lot easier to provide you with good feedback:

- The ideal bug report contains a [short reproducible code snippet](https://scikit-learn.org/stable/developers/minimal_reproducer.html#minimal-reproducer), this way anyone can try to reproduce the bug easily. If your snippet is longer than around 50 lines, please link to a [Gist](https://gist.github.com) or a GitHub repo.

- If not feasible to include a reproducible snippet, please be specific about what **estimators and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as your **Python, scikit-learn, numpy, and scipy versions**. This information can be found by running:

  ```
  python -c "import sklearn; sklearn.show_versions()"
  ```

- Please ensure all **code snippets and error messages are formatted in appropriate code blocks**. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks) for more details.

- Please be explicit **how this issue impacts you as a scikit-learn user**. Giving some details (a short paragraph) about how you use scikit-learn and why you need this issue resolved will help the project maintainers invest time and effort on issues that actually impact users.

- Please tell us if you would be interested in opening a PR to resolve your issue once triaged by a project maintainer.

Note that the scikit-learn tracker receives [daily reports](https://github.com/scikit-learn/scikit-learn/issues?q=label%3Aspam) by GitHub accounts that are mostly interested in increasing contribution statistics and show little interest in the expected end-user impact of their contributions. As project maintainers we want to be able to assess if our efforts are likely to have a meaningful and positive impact to our end users. Therefore, we ask you to avoid opening issues for things you don’t actually care about.

If you want to help curate issues, read about [Bug triaging and issue curation](https://scikit-learn.org/stable/developers/bug_triaging.html#bug-triaging).

### Contributing code and documentation

The preferred way to contribute to scikit-learn is to fork the [main repository](https://github.com/scikit-learn/scikit-learn/) on GitHub, then submit a “pull request” (PR).

To get started, you need to

1. [Set up your development environment](https://scikit-learn.org/stable/developers/development_setup.html#setup-development-environment)

1. Find an issue to work on (see [New Contributors](https://scikit-learn.org/stable/developers/contributing.html#new-contributors))

1. Follow the [Development workflow](https://scikit-learn.org/stable/developers/contributing.html#development-workflow)

1. Make sure, you noted the [Pull request checklist](https://scikit-learn.org/stable/developers/contributing.html#pr-checklist)

If you want to contribute [Documentation](https://scikit-learn.org/stable/developers/contributing.html#contribute-documentation), make sure you are able to [build it locally](https://scikit-learn.org/stable/developers/contributing.html#building-documentation), before submitting a PR.

Note

To avoid duplicating work, it is highly advised that you search through the [issue tracker](https://github.com/scikit-learn/scikit-learn/issues) and the [PR list](https://github.com/scikit-learn/scikit-learn/pulls). If in doubt about duplicated work, or if you want to work on a non-trivial feature, it’s recommended to first open an issue in the [issue tracker](https://github.com/scikit-learn/scikit-learn/issues) to get some feedback from core developers.

One easy way to find an issue to work on is by applying the “help wanted” label in your search. This lists all the issues that have been unclaimed so far. If you’d like to work on such issue, leave a comment with your idea of how you plan to approach it, and start working on it. If somebody else has already said they’d be working on the issue in the past 2-3 weeks, please let them finish their work, otherwise consider it stalled and take it over.

To maintain the quality of the codebase and ease the review process, any contribution must conform to the project’s [coding guidelines](https://scikit-learn.org/stable/developers/develop.html#coding-guidelines), in particular:

- Don’t modify unrelated lines to keep the PR focused on the scope stated in its description or issue.

- Only write inline comments that add value and avoid stating the obvious: explain the “why” rather than the “what”.

- **Most importantly**: Do not contribute code that you don’t understand.

#### Development workflow

The next steps describe the process of modifying code and submitting a PR:

1. Synchronize your `main` branch with the `upstream/main` branch, more details on [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork):

   ```
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

1. Create a feature branch to hold your development changes:

   ```
   git checkout -b my_feature
   ```

   and start making changes. Always use a feature branch. It’s good practice to never work on the `main` branch!

1. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using `git add` and then `git commit`:

   ```
   git add modified_files
   git commit
   ```

   Note

   [pre-commit](https://scikit-learn.org/stable/developers/development_setup.html#pre-commit) may reformat your code automatically when you do `git commit`. When this happens, you need to do `git add` followed by `git commit` again. In some rarer cases, you may need to fix things manually, use the error message to figure out what needs to be changed, and use `git add` followed by `git commit` until the commit is successful.

   Then push the changes to your GitHub account with:

   ```
   git push -u origin my_feature
   ```

1. Follow [these](https://help.github.com/articles/creating-a-pull-request-from-a-fork) instructions to create a pull request from your fork. This will send a notification to potential reviewers. You may want to consider sending a message to the [discord](https://discord.com/invite/h9qyrK8Jc8) in the development channel for more visibility if your pull request does not receive attention after a couple of days (instant replies are not guaranteed though).

It is often helpful to keep your local feature branch synchronized with the latest changes of the main scikit-learn repository:

```
git fetch upstream
git merge upstream/main
```

Subsequently, you might need to solve the conflicts. You can refer to the [Git documentation related to resolving merge conflict using the command line](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/).

Learning Git

The [Git documentation](https://git-scm.com/doc) and [https://try.github.io](https://try.github.io) are excellent resources to get started with git, and understanding all of the commands shown here.

#### Pull request checklist

Before a PR can be merged, it needs to be approved by two core developers. An incomplete contribution – where you expect to do more work before receiving a full review – should be marked as a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request) and changed to “ready for review” when it matures. Draft PRs may be useful to: indicate you are working on something to avoid duplicated work, request broad review of functionality or API, or seek collaborators. Draft PRs often benefit from the inclusion of a [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments) in the PR description.

In order to ease the reviewing process, we recommend that your contribution complies with the following rules before marking a PR as “ready for review”. The **bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarizes what your contribution does. This title will often become the commit message once merged so it should summarize your contribution for posterity. In some cases “Fix \<ISSUE TITLE>” is enough. “Fix #\<ISSUE NUMBER>” is never a good title.

1. **Pull requests are expected to resolve one or more issues**. Please **do not open PRs for issues that are labeled as “Needs triage”** (see [Issues tagged “Needs Triage”](https://scikit-learn.org/stable/developers/contributing.html#issues-tagged-needs-triage)) or with other kinds of “Needs …” labels. Please do not open PRs for issues for which:

   - the discussion has not settled down to an explicit resolution plan,

   - the reporter has already expressed interest in opening a PR,

   - there already exists cross-referenced and active PRs.

   If merging your pull request means that some other issues/PRs should be closed, you should [use keywords to create link to them](https://github.com/blog/1506-closing-issues-via-pull-requests/) (e.g., `Fixes #1234`; multiple issues/PRs are allowed as long as each one is preceded by a keyword). Upon merging, those issues/PRs will automatically be closed by GitHub. If your pull request is simply related to some other issues/PRs, or it only partially resolves the target issue, create a link to them without using the keywords (e.g., `Towards #1234`).

1. **Make sure your code passes the tests**. The whole test suite can be run with `pytest`, but it is usually not recommended since it takes a long time. It is often enough to only run the test related to your changes: for example, if you changed something in `sklearn/linear_model/_logistic.py`, running the following commands will usually be enough:

   - `pytest sklearn/linear_model/_logistic.py` to make sure the doctest examples are correct

   - `pytest sklearn/linear_model/tests/test_logistic.py` to run the tests specific to the file

   - `pytest sklearn/linear_model` to test the whole [`linear_model`](https://scikit-learn.org/stable/api/sklearn.linear_model.html#module-sklearn.linear_model "sklearn.linear_model") module

   - `pytest doc/modules/linear_model.rst` to make sure the user guide examples are correct.

   - `pytest sklearn/tests/test_common.py -k LogisticRegression` to run all our estimator checks (specifically for `LogisticRegression`, if that’s the estimator you changed).

   There may be other failing tests, but they will be caught by the CI so you don’t need to run the whole test suite locally. For guidelines on how to use `pytest` efficiently, see the [Useful pytest aliases and flags](https://scikit-learn.org/stable/developers/tips.html#pytest-tips).

1. **Make sure your code is properly commented and documented**, and **make sure the documentation renders properly**. To build the documentation, please refer to our [Documentation](https://scikit-learn.org/stable/developers/contributing.html#contribute-documentation) guidelines. The CI will also build the docs: please refer to [Generated documentation on GitHub Actions](https://scikit-learn.org/stable/developers/contributing.html#generated-doc-ci).

1. **Tests are necessary for enhancements to be accepted**. Bug-fixes or new features should be provided with non-regression tests. These tests verify the correct behavior of the fix or feature. In this manner, further modifications on the code base are granted to be consistent with the desired behavior. In the case of bug fixes, at the time of the PR, the non-regression tests should fail for the code base in the `main` branch and pass for the PR code.

1. If your PR is likely to affect users, you need to add a changelog entry describing your PR changes. See the [README](https://github.com/scikit-learn/scikit-learn/blob/main/doc/whats_new/upcoming_changes/README.md) for more details.

1. Follow the [Coding guidelines](https://scikit-learn.org/stable/developers/develop.html#coding-guidelines).

1. When applicable, use the validation tools and scripts in the [`sklearn.utils`](https://scikit-learn.org/stable/api/sklearn.utils.html#module-sklearn.utils "sklearn.utils") module. A list of utility routines available for developers can be found in the [Utilities for Developers](https://scikit-learn.org/stable/developers/utilities.html#developers-utils) page.

1. PRs should often substantiate the change, through benchmarks of performance and efficiency (see [Monitoring performance](https://scikit-learn.org/stable/developers/contributing.html#monitoring-performances)) or through examples of usage. Examples also illustrate the features and intricacies of the library to users. Have a look at other examples in the [examples/](https://github.com/scikit-learn/scikit-learn/tree/main/examples) directory for reference. Examples should demonstrate why the new functionality is useful in practice and, if possible, compare it to other methods available in scikit-learn.

1. New features have some maintenance overhead. We expect PR authors to take part in the maintenance for the code they submit, at least initially. New features need to be illustrated with narrative documentation in the user guide, with small code snippets. If relevant, please also add references in the literature, with PDF links when possible.

1. The user guide should also include expected time and space complexity of the algorithm and scalability, e.g. “this algorithm can scale to a large number of samples > 100000, but does not scale in dimensionality: `n_features` is expected to be lower than 100”.

You can also check our [Code Review Guidelines](https://scikit-learn.org/stable/developers/contributing.html#code-review) to get an idea of what reviewers will expect.

You can check for common programming errors with the following tools:

- Code with a good unit test coverage (at least 80%, better 100%), check with:

  ```
  pip install pytest pytest-cov
  pytest --cov sklearn path/to/tests
  ```

  See also [Testing and improving test coverage](https://scikit-learn.org/stable/developers/contributing.html#testing-coverage).

- Run static analysis with `mypy`:

  ```
  mypy sklearn
  ```

  This must not produce new errors in your pull request. Using `# type: ignore` annotation can be a workaround for a few cases that are not supported by mypy, in particular,

  - when importing C or Cython modules,

  - on properties with decorators.

Bonus points for contributions that include a performance analysis with a benchmark script and profiling output (see [Monitoring performance](https://scikit-learn.org/stable/developers/contributing.html#monitoring-performances)). Also check out the [How to optimize for speed](https://scikit-learn.org/stable/developers/performance.html#performance-howto) guide for more details on profiling and Cython optimizations.

Note

The current state of the scikit-learn code base is not compliant with all of those guidelines, but we expect that enforcing those constraints on all new contributions will get the overall code base quality in the right direction.

See also

For two very well documented and more detailed guides on development workflow, please pay a visit to the [Scipy Development Workflow](https://scipy.github.io/devdocs/dev/dev_quickstart.html) - and the [Astropy Workflow for Developers](https://astropy.readthedocs.io/en/latest/development/workflow/development_workflow.html) sections.

#### Continuous Integration (CI)

- Github Actions are used for various tasks, including testing scikit-learn on Linux, Mac and Windows, with different dependencies and settings, building wheels and source distributions.

- CircleCI is used to build the docs for viewing.

##### Commit message markers

Please note that if one of the following markers appears in the latest commit message, the following actions are taken.

| Commit Message Marker | Action Taken by CI                                                                                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ci skip]             | CI is skipped completely                                                                                                                                                                       |
| [cd build]            | CD is run (wheels and source distribution are built)                                                                                                                                           |
| [scipy-dev]           | Build & test with our dependencies (numpy, scipy, etc.) development builds                                                                                                                     |
| [free-threaded]       | Build & test with CPython 3.14 free-threaded                                                                                                                                                   |
| [pyodide]             | Build & test with Pyodide                                                                                                                                                                      |
| [float32]             | Run float32 tests by setting `SKLEARN_RUN_FLOAT32_TESTS=1`. See [Environment variables](https://scikit-learn.org/stable/computing/parallelism.html#environment-variable) for more details      |
| [all random seeds]    | Run tests using the `global_random_seed` fixture with all random seeds. See [this](https://github.com/scikit-learn/scikit-learn/issues/28959) for more details about the commit message format |
| [doc skip]            | Docs are not built                                                                                                                                                                             |
| [doc quick]           | Docs built, but excludes example gallery plots                                                                                                                                                 |
| [doc build]           | Docs built including example gallery plots (very long)                                                                                                                                         |

Note that, by default, the documentation is built but only the examples that are directly modified by the pull request are executed.

##### Resolve conflicts in lock files

Here is a bash snippet that helps resolving conflicts in environment and lock files:

```
# pull latest upstream/main
git pull upstream main --no-rebase
# resolve conflicts - keeping the upstream/main version for specific files
git checkout --theirs  build_tools/*/*.lock build_tools/*/*environment.yml \
    build_tools/*/*lock.txt build_tools/*/*requirements.txt
git add build_tools/*/*.lock build_tools/*/*environment.yml \
    build_tools/*/*lock.txt build_tools/*/*requirements.txt
git merge --continue
```

This will merge `upstream/main` into our branch, automatically prioritising the `upstream/main` for conflicting environment and lock files (this is good enough, because we will re-generate the lock files afterwards).

Note that this only fixes conflicts in environment and lock files and you might have other conflicts to resolve.

Finally, we have to re-generate the environment and lock files for the CIs by running:

```
python build_tools/update_environments_and_lock_files.py
```

#### Stalled pull requests

As contributing a feature can be a lengthy process, some pull requests appear inactive but unfinished. In such a case, taking them over is a great service for the project. A good etiquette to take over is:

- **Determine if a PR is stalled**

  - A pull request may have the label “stalled” or “help wanted” if we have already identified it as a candidate for other contributors.

  - To decide whether an inactive PR is stalled, ask the contributor if she/he plans to continue working on the PR in the near future. Failure to respond within 2 weeks with an activity that moves the PR forward suggests that the PR is stalled and will result in tagging that PR with “help wanted”.

    Note that if a PR has received earlier comments on the contribution that have had no reply in a month, it is safe to assume that the PR is stalled and to shorten the wait time to one day.

    After a sprint, follow-up for un-merged PRs opened during sprint will be communicated to participants at the sprint, and those PRs will be tagged “sprint”. PRs tagged with “sprint” can be reassigned or declared stalled by sprint leaders.

- **Taking over a stalled PR**: To take over a PR, it is important to comment on the stalled PR that you are taking over and to link from the new PR to the old one. The new PR should be created by pulling from the old one.

#### Stalled and Unclaimed Issues

Generally speaking, issues which are up for grabs will have a [“help wanted”](https://github.com/scikit-learn/scikit-learn/labels/help%20wanted). tag. However, not all issues which need contributors will have this tag, as the “help wanted” tag is not always up-to-date with the state of the issue. Contributors can find issues which are still up for grabs using the following guidelines:

- First, to **determine if an issue is claimed**:

  - Check for linked pull requests

  - Check the conversation to see if anyone has said that they’re working on creating a pull request

- If a contributor comments on an issue to say they are working on it, a pull request is expected within 2 weeks (new contributor) or 4 weeks (contributor or core dev), unless a larger time frame is explicitly given. Beyond that time, another contributor can take the issue and make a pull request for it. We encourage contributors to comment directly on the stalled or unclaimed issue to let community members know that they will be working on it.

- If the issue is linked to a [stalled pull request](https://scikit-learn.org/stable/developers/contributing.html#stalled-pull-request), we recommend that contributors follow the procedure described in the [Stalled pull requests](https://scikit-learn.org/stable/developers/contributing.html#stalled-pull-request) section rather than working directly on the issue.

#### Issues tagged “Needs Triage”

The [“Needs Triage”](https://github.com/scikit-learn/scikit-learn/labels/needs%20triage) label means that the issue is not yet confirmed or fully understood. It signals to scikit-learn members to clarify the problem, discuss scope, and decide on the next steps. You are welcome to join the discussion, but as per our [Code of Conduct](https://github.com/scikit-learn/scikit-learn/blob/main/CODE_OF_CONDUCT.md) please do not open a PR until the “Needs Triage” label is removed, there is a clear consensus on addressing the issue and some directions on how to address it.

#### Video resources

These videos are step-by-step introductions on how to contribute to scikit-learn, and are a great companion to the text guidelines. Please make sure to still check our guidelines, since they describe our latest up-to-date workflow.

- Crash Course in Contributing to Scikit-Learn & Open Source Projects: [Video](https://youtu.be/5OL8XoMMOfA), [Transcript](https://github.com/data-umbrella/event-transcripts/blob/main/2020/05-andreas-mueller-contributing.md)

- Example of Submitting a Pull Request to scikit-learn: [Video](https://youtu.be/PU1WyDPGePI), [Transcript](https://github.com/data-umbrella/event-transcripts/blob/main/2020/06-reshama-shaikh-sklearn-pr.md)

- Sprint-specific instructions and practical tips: [Video](https://youtu.be/p_2Uw2BxdhA), [Transcript](https://github.com/data-umbrella/data-umbrella-scikit-learn-sprint/blob/master/3_transcript_ACM_video_vol2.md)

- 3 Components of Reviewing a Pull Request: [Video](https://youtu.be/dyxS9KKCNzA), [Transcript](https://github.com/data-umbrella/event-transcripts/blob/main/2021/27-thomas-pr.md)

Note

In January 2021, the default branch name changed from `master` to `main` for the scikit-learn GitHub repository to use more inclusive terms. These videos were created prior to the renaming of the branch. For contributors who are viewing these videos to set up their working environment and submitting a PR, `master` should be replaced to `main`.

### Documentation

We welcome thoughtful contributions to the documentation and are happy to review additions in the following areas:

- **Function/method/class docstrings:** Also known as “API documentation”, these describe what the object does and detail any parameters, attributes and methods. Docstrings live alongside the code in [sklearn/](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn), and are generated according to [doc/api_reference.py](https://github.com/scikit-learn/scikit-learn/blob/main/doc/api_reference.py). To add, update, remove, or deprecate a public API that is listed in [API Reference](https://scikit-learn.org/stable/api/index.html#api-ref), this is the place to look at.

- **User guide:** These provide more detailed information about the algorithms implemented in scikit-learn and generally live in the root [doc/](https://github.com/scikit-learn/scikit-learn/tree/main/doc) directory and [doc/modules/](https://github.com/scikit-learn/scikit-learn/tree/main/doc/modules).

- **Examples:** These provide full code examples that may demonstrate the use of scikit-learn modules, compare different algorithms or discuss their interpretation, etc. Examples live in [examples/](https://github.com/scikit-learn/scikit-learn/tree/main/examples).

- **Other reStructuredText documents:** These provide various other useful information (e.g., the [Contributing to pandas](https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#contributing "(in pandas v3.0.3)") guide) and live in [doc/](https://github.com/scikit-learn/scikit-learn/tree/main/doc).

Guidelines for writing docstrings

- You can use `pytest` to test docstrings, e.g. assuming the `RandomForestClassifier` docstring has been modified, the following command would test its docstring compliance:

  ```
  pytest --doctest-modules sklearn/ensemble/_forest.py -k RandomForestClassifier
  ```

- The correct order of sections is: Parameters, Returns, See Also, Notes, Examples. See the [numpydoc documentation](https://numpydoc.readthedocs.io/en/latest/format.html#sections) for information on other possible sections.

- When documenting the parameters and attributes, here is a list of some well-formatted examples

  ```
  n_clusters : int, default=3
      The number of clusters detected by the algorithm.

  some_param : {"hello", "goodbye"}, bool or int, default=True
      The parameter description goes here, which can be either a string
      literal (either `hello` or `goodbye`), a bool, or an int. The default
      value is True.

  array_parameter : {array-like, sparse matrix} of shape (n_samples, n_features) \
      or (n_samples,)
      This parameter accepts data in either of the mentioned forms, with one
      of the mentioned shapes. The default value is `np.ones(shape=(n_samples,))`.

  list_param : list of int

  typed_ndarray : ndarray of shape (n_samples,), dtype=np.int32

  sample_weight : array-like of shape (n_samples,), default=None

  multioutput_array : ndarray of shape (n_samples, n_classes) or list of such arrays
  ```

  In general have the following in mind:

  - Use Python basic types. (`bool` instead of `boolean`)

  - Use parenthesis for defining shapes: `array-like of shape (n_samples,)` or `array-like of shape (n_samples, n_features)`

  - For strings with multiple options, use brackets: `input: {'log', 'squared', 'multinomial'}`

  - 1D or 2D data can be a subset of `{array-like, ndarray, sparse matrix, dataframe}`. Note that `array-like` can also be a `list`, while `ndarray` is explicitly only a `numpy.ndarray`.

  - Specify `dataframe` when “frame-like” features are being used, such as the column names.

  - When specifying the data type of a list, use `of` as a delimiter: `list of int`. When the parameter supports arrays giving details about the shape and/or data type and a list of such arrays, you can use one of `array-like of shape (n_samples,) or list of such arrays`.

  - When specifying the dtype of an ndarray, use e.g. `dtype=np.int32` after defining the shape: `ndarray of shape (n_samples,), dtype=np.int32`. You can specify multiple dtype as a set: `array-like of shape (n_samples,), dtype={np.float64, np.float32}`. If one wants to mention arbitrary precision, use `integral` and `floating` rather than the Python dtype `int` and `float`. When both `int` and `floating` are supported, there is no need to specify the dtype.

  - When the default is `None`, `None` only needs to be specified at the end with `default=None`. Be sure to include in the docstring, what it means for the parameter or attribute to be `None`.

- Add “See Also” in docstrings for related classes/functions.

- “See Also” in docstrings should be one line per reference, with a colon and an explanation, for example:

  ```
  See Also
  --------
  SelectKBest : Select features based on the k highest scores.
  SelectFpr : Select features based on a false positive rate test.
  ```

- The “Notes” section is optional. It is meant to provide information on specific behavior of a function/class/classmethod/method.

- A `Note` can also be added to an attribute, but in that case it requires using the `.. rubric:: Note` directive.

- Add one or two **snippets** of code in “Example” section to show how it can be used. The code should be runnable as is, i.e. it should include all required imports. Keep this section as brief as possible.

Guidelines for writing the user guide and other reStructuredText documents

It is important to keep a good compromise between mathematical and algorithmic details, and give intuition to the reader on what the algorithm does.

- Begin with a concise, hand-waving explanation of what the algorithm/code does on the data.

- Highlight the usefulness of the feature and its recommended application. Consider including the algorithm’s complexity (\\O\\left(g\\left(n\\right)\\right)\\) if available, as “rules of thumb” can be very machine-dependent. Only if those complexities are not available, then rules of thumb may be provided instead.

- Incorporate a relevant figure (generated from an example) to provide intuitions.

- Include one or two short code examples to demonstrate the feature’s usage.

- Introduce any necessary mathematical equations, followed by references. By deferring the mathematical aspects, the documentation becomes more accessible to users primarily interested in understanding the feature’s practical implications rather than its underlying mechanics.

- When editing reStructuredText (`.rst`) files, try to keep line length under 88 characters when possible (exceptions include links and tables).

- In scikit-learn reStructuredText files both single and double backticks surrounding text will render as inline literal (often used for code, e.g., `list`). This is due to specific configurations we have set. Single backticks should be used nowadays.

- Too much information makes it difficult for users to access the content they are interested in. Use dropdowns to factorize it by using the following syntax

  ```
  .. dropdown:: Dropdown title

    Dropdown content.
  ```

  The snippet above will result in the following dropdown:

  Dropdown title

  Dropdown content.

- Information that can be hidden by default using dropdowns is:

  - low hierarchy sections such as `References`, `Properties`, etc. (see for instance the subsections in [Detection error tradeoff (DET)](https://scikit-learn.org/stable/modules/model_evaluation.html#det-curve));

  - in-depth mathematical details;

  - narrative that is use-case specific;

  - in general, narrative that may only interest users that want to go beyond the pragmatics of a given tool.

- Do not use dropdowns for the low level section `Examples`, as it should stay visible to all users. Make sure that the `Examples` section comes right after the main discussion with the least possible folded section in-between.

- Be aware that dropdowns break cross-references. If that makes sense, hide the reference along with the text mentioning it. Else, do not use dropdown.

Guidelines for writing references

- When bibliographic references are available with [arxiv](https://arxiv.org/) or [Digital Object Identifier](https://www.doi.org/) identification numbers, use the sphinx directives `:arxiv:` or `:doi:`. For example, see references in [Spectral Clustering Graphs](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering-graph).

- For the “References” section in docstrings, see [`sklearn.metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score "sklearn.metrics.silhouette_score") as an example.

- To cross-reference to other pages in the scikit-learn documentation use the reStructuredText cross-referencing syntax:

  - **Section:** to link to an arbitrary section in the documentation, use reference labels (see [Sphinx docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#ref-role)). For example:

    ```
    .. _my-section:

    My section
    ----------

    This is the text of the section.

    To refer to itself use :ref:`my-section`.
    ```

    You should not modify existing sphinx reference labels as this would break existing cross references and external links pointing to specific sections in the scikit-learn documentation.

  - **Glossary:** linking to a term in the [Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html#glossary):

    ```
    :term:`cross_validation`
    ```

  - **Function:** to link to the documentation of a function, use the full import path to the function:

    ```
    :func:`~sklearn.model_selection.cross_val_score`
    ```

    However, if there is a `.. currentmodule::` directive above you in the document, you will only need to use the path to the function succeeding the current module specified. For example:

    ```
    .. currentmodule:: sklearn.model_selection

    :func:`cross_val_score`
    ```

  - **Class:** to link to documentation of a class, use the full import path to the class, unless there is a `.. currentmodule::` directive in the document above (see above):

    ```
    :class:`~sklearn.preprocessing.StandardScaler`
    ```

You can edit the documentation using any text editor, and then generate the HTML output by following [Building the documentation](https://scikit-learn.org/stable/developers/contributing.html#building-documentation). The resulting HTML files will be placed in `_build/html/` and are viewable in a web browser, for instance by opening the local `_build/html/index.html` file or by running a local server

```
python -m http.server -d _build/html
```

#### Building the documentation

**Before submitting a pull request check if your modifications have introduced new sphinx warnings by building the documentation locally and try to fix them.**

First, make sure you have [properly installed](https://scikit-learn.org/stable/developers/development_setup.html#setup-development-environment) the development version. On top of that, building the documentation requires installing some additional packages:

```
pip install sphinx sphinx-gallery numpydoc matplotlib Pillow pandas \
            polars scikit-image packaging seaborn sphinx-prompt \
            sphinxext-opengraph sphinx-copybutton plotly pooch \
            pydata-sphinx-theme sphinxcontrib-sass sphinx-design \
            sphinx-remove-toctrees
```

To build the documentation, you need to be in the `doc` folder:

```
cd doc
```

In the vast majority of cases, you only need to generate the web site without the example gallery:

```
make
```

The documentation will be generated in the `_build/html/stable` directory and are viewable in a web browser, for instance by opening the local `_build/html/stable/index.html` file. To also generate the example gallery you can use:

```
make html
```

This will run all the examples, which takes a while. You can also run only a few examples based on their file names. Here is a way to run all examples with filenames containing `plot_calibration`:

```
EXAMPLES_PATTERN="plot_calibration" make html
```

You can use regular expressions for more advanced use cases.

Set the environment variable `NO_MATHJAX=1` if you intend to view the documentation in an offline setting. To build the PDF manual, run:

```
make latexpdf
```

Sphinx version

While we do our best to have the documentation build under as many versions of Sphinx as possible, the different versions tend to behave slightly differently. To get the best results, you should use the same version as the one we used on CircleCI. Look at this [GitHub search](https://github.com/search?q=repo%3Ascikit-learn%2Fscikit-learn+%2F%5C%2Fsphinx-%5B0-9.%5D%2B%2F+path%3Abuild_tools%2Fcircle%2Fdoc_linux-64_conda.lock&type=code) to know the exact version.

#### Generated documentation on GitHub Actions

When you change the documentation in a pull request, GitHub Actions automatically builds it. To view the documentation generated by GitHub Actions, simply go to the bottom of your PR page, look for the item “Check the rendered docs here!” and click on ‘details’ next to it:

![../images/generated-doc-ci.png](https://scikit-learn.org/stable/_images/generated-doc-ci.png)

### Testing and improving test coverage

High-quality [unit testing](https://en.wikipedia.org/wiki/Unit_testing) is a corner-stone of the scikit-learn development process. For this purpose, we use the [pytest](https://docs.pytest.org) package. The tests are functions appropriately named, located in `tests` subdirectories, that check the validity of the algorithms and the different options of the code.

Running `pytest` in a folder will run all the tests of the corresponding subpackages. For a more detailed `pytest` workflow, please refer to the [Pull request checklist](https://scikit-learn.org/stable/developers/contributing.html#pr-checklist).

We expect code coverage of new features to be at least around 90%.

Writing matplotlib-related tests

Test fixtures ensure that a set of tests will be executing with the appropriate initialization and cleanup. The scikit-learn test suite implements a `pyplot` fixture which can be used with `matplotlib`.

The `pyplot` fixture should be used when a test function is dealing with `matplotlib`. `matplotlib` is a soft dependency and is not required. This fixture is in charge of skipping the tests if `matplotlib` is not installed. In addition, figures created during the tests will be automatically closed once the test function has been executed.

To use this fixture in a test function, one needs to pass it as an argument:

```
def test_requiring_mpl_fixture(pyplot):
    # you can now safely use matplotlib
```

Workflow to improve test coverage

To test code coverage, you need to install the [coverage](https://pypi.org/project/coverage/) package in addition to `pytest`.

1. Run `pytest --cov sklearn /path/to/tests`. The output lists for each file the line numbers that are not tested.

1. Find a low hanging fruit, looking at which lines are not tested, write or adapt a test specifically for these lines.

1. Loop.

### Monitoring performance

*This section is heavily inspired from the* [pandas documentation](https://pandas.pydata.org/docs/development/contributing_codebase.html#running-the-performance-test-suite).

When proposing changes to the existing code base, it’s important to make sure that they don’t introduce performance regressions. Scikit-learn uses [asv benchmarks](https://github.com/airspeed-velocity/asv) to monitor the performance of a selection of common estimators and functions. You can view these benchmarks on the [scikit-learn benchmark page](https://scikit-learn.org/scikit-learn-benchmarks). The corresponding benchmark suite can be found in the `asv_benchmarks/` directory.

To use all features of asv, you will need either `conda` or `virtualenv`. For more details please check the [asv installation webpage](https://asv.readthedocs.io/en/latest/installing.html).

First of all you need to install the development version of asv:

```
pip install git+https://github.com/airspeed-velocity/asv
```

and change your directory to `asv_benchmarks/`:

```
cd asv_benchmarks
```

The benchmark suite is configured to run against your local clone of scikit-learn. Make sure it is up to date:

```
git fetch upstream
```

In the benchmark suite, the benchmarks are organized following the same structure as scikit-learn. For example, you can compare the performance of a specific estimator between `upstream/main` and the branch you are working on:

```
asv continuous -b LogisticRegression upstream/main HEAD
```

The command uses conda by default for creating the benchmark environments. If you want to use virtualenv instead, use the `-E` flag:

```
asv continuous -E virtualenv -b LogisticRegression upstream/main HEAD
```

You can also specify a whole module to benchmark:

```
asv continuous -b linear_model upstream/main HEAD
```

You can replace `HEAD` by any local branch. By default it will only report the benchmarks that have changed by at least 10%. You can control this ratio with the `-f` flag.

To run the full benchmark suite, simply remove the `-b` flag :

```
asv continuous upstream/main HEAD
```

However this can take up to two hours. The `-b` flag also accepts a regular expression for a more complex subset of benchmarks to run.

To run the benchmarks without comparing to another branch, use the `run` command:

```
asv run -b linear_model HEAD^!
```

You can also run the benchmark suite using the version of scikit-learn already installed in your current Python environment:

```
asv run --python=same
```

It’s particularly useful when you installed scikit-learn in editable mode to avoid creating a new environment each time you run the benchmarks. By default the results are not saved when using an existing installation. To save the results you must specify a commit hash:

```
asv run --python=same --set-commit-hash=<commit hash>
```

Benchmarks are saved and organized by machine, environment and commit. To see the list of all saved benchmarks:

```
asv show
```

and to see the report of a specific run:

```
asv show <commit hash>
```

When running benchmarks for a pull request you’re working on please report the results on github.

The benchmark suite supports additional configurable options which can be set in the `benchmarks/config.json` configuration file. For example, the benchmarks can run for a provided list of values for the `n_jobs` parameter.

More information on how to write a benchmark and how to use asv can be found in the [asv documentation](https://asv.readthedocs.io/en/latest/index.html).

### Issue Tracker Tags

All issues and pull requests on the [GitHub issue tracker](https://github.com/scikit-learn/scikit-learn/issues) should have (at least) one of the following tags:

Bug:
Something is happening that clearly shouldn’t happen. Wrong results as well as unexpected errors from estimators go here.

Enhancement:
Improving performance, usability, consistency.

Documentation:
Missing, incorrect or sub-standard documentations and examples.

New Feature:
Feature requests and pull requests implementing a new feature.

There are four other tags to help new contributors:

Good first issue:
This issue is ideal for a first contribution to scikit-learn. Ask for help if the formulation is unclear. If you have already contributed to scikit-learn, look at Easy issues instead.

Easy:
This issue can be tackled without much prior experience.

Moderate:
Might need some knowledge of machine learning or the package, but is still approachable for someone new to the project.

Help wanted:
This tag marks an issue which currently lacks a contributor or a PR that needs another contributor to take over the work. These issues can range in difficulty, and may not be approachable for new contributors. Note that not all issues which need contributors will have this tag.

### Maintaining backwards compatibility

#### Deprecation

If any publicly accessible class, function, method, attribute or parameter is renamed, we still support the old one for two releases and issue a deprecation warning when it is called, passed, or accessed.

Deprecating a class or a function

Suppose the function `zero_one` is renamed to `zero_one_loss`, we add the decorator [`utils.deprecated`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated "sklearn.utils.deprecated") to `zero_one` and call `zero_one_loss` from that function:

```
from sklearn.utils import deprecated

def zero_one_loss(y_true, y_pred, normalize=True):
    # actual implementation
    pass

@deprecated(
    "Function `zero_one` was renamed to `zero_one_loss` in 0.13 and will be "
    "removed in 0.15. Default behavior is changed from `normalize=False` to "
    "`normalize=True`"
)
def zero_one(y_true, y_pred, normalize=False):
    return zero_one_loss(y_true, y_pred, normalize)
```

One also needs to move `zero_one` from `API_REFERENCE` to `DEPRECATED_API_REFERENCE` and add `zero_one_loss` to `API_REFERENCE` in the `doc/api_reference.py` file to reflect the changes in [API Reference](https://scikit-learn.org/stable/api/index.html#api-ref).

Deprecating an attribute or a method

If an attribute or a method is to be deprecated, use the decorator [`deprecated`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated "sklearn.utils.deprecated") on the property. Please note that the [`deprecated`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated "sklearn.utils.deprecated") decorator should be placed before the `property` decorator if there is one, so that the docstrings can be rendered properly. For instance, renaming an attribute `labels_` to `classes_` can be done as:

```
@deprecated(
    "Attribute `labels_` was deprecated in 0.13 and will be removed in 0.15. Use "
    "`classes_` instead"
)
@property
def labels_(self):
    return self.classes_
```

Deprecating a parameter

If a parameter has to be deprecated, a `FutureWarning` warning must be raised manually. In the following example, `k` is deprecated and renamed to n_clusters:

```
import warnings

def example_function(n_clusters=8, k="deprecated"):
    if k != "deprecated":
        warnings.warn(
            "`k` was renamed to `n_clusters` in 0.13 and will be removed in 0.15",
            FutureWarning,
        )
        n_clusters = k
```

When the change is in a class, we validate and raise warning in `fit`:

```
import warnings

class ExampleEstimator(BaseEstimator):
    def __init__(self, n_clusters=8, k='deprecated'):
        self.n_clusters = n_clusters
        self.k = k

    def fit(self, X, y):
        if self.k != "deprecated":
            warnings.warn(
                "`k` was renamed to `n_clusters` in 0.13 and will be removed in 0.15.",
                FutureWarning,
            )
            self._n_clusters = self.k
        else:
            self._n_clusters = self.n_clusters
```

As in these examples, the warning message should always give both the version in which the deprecation happened and the version in which the old behavior will be removed. If the deprecation happened in version 0.x-dev, the message should say deprecation occurred in version 0.x and the removal will be in 0.(x+2), so that users will have enough time to adapt their code to the new behaviour. For example, if the deprecation happened in version 0.18-dev, the message should say it happened in version 0.18 and the old behavior will be removed in version 0.20.

The warning message should also include a brief explanation of the change and point users to an alternative.

In addition, a deprecation note should be added in the docstring, recalling the same information as the deprecation warning as explained above. Use the `.. deprecated::` directive:

```
.. deprecated:: 0.13
   ``k`` was renamed to ``n_clusters`` in version 0.13 and will be removed
   in 0.15.
```

What’s more, a deprecation requires a test which ensures that the warning is raised in relevant cases but not in other cases. The warning should be caught in all other tests (using e.g., `@pytest.mark.filterwarnings`), and there should be no warning in the examples.

#### Change the default value of a parameter

If the default value of a parameter needs to be changed, please replace the default value with a specific value (e.g., `"warn"`) and raise `FutureWarning` when users are using the default value. The following example assumes that the current version is 0.20 and that we change the default value of `n_clusters` from 5 (old default for 0.20) to 10 (new default for 0.22):

```
import warnings

def example_function(n_clusters="warn"):
    if n_clusters == "warn":
        warnings.warn(
            "The default value of `n_clusters` will change from 5 to 10 in 0.22.",
            FutureWarning,
        )
        n_clusters = 5
```

When the change is in a class, we validate and raise warning in `fit`:

```
import warnings

class ExampleEstimator:
    def __init__(self, n_clusters="warn"):
        self.n_clusters = n_clusters

    def fit(self, X, y):
        if self.n_clusters == "warn":
            warnings.warn(
                "The default value of `n_clusters` will change from 5 to 10 in 0.22.",
                FutureWarning,
            )
            self._n_clusters = 5
```

Similar to deprecations, the warning message should always give both the version in which the change happened and the version in which the old behavior will be removed.

The parameter description in the docstring needs to be updated accordingly by adding a `versionchanged` directive with the old and new default value, pointing to the version when the change will be effective:

```
.. versionchanged:: 0.22
   The default value for `n_clusters` will change from 5 to 10 in version 0.22.
```

Finally, we need a test which ensures that the warning is raised in relevant cases but not in other cases. The warning should be caught in all other tests (using e.g., `@pytest.mark.filterwarnings`), and there should be no warning in the examples.

### Code Review Guidelines

Reviewing code contributed to the project as PRs is a crucial component of scikit-learn development. We encourage anyone to start reviewing code of other developers. The code review process is often highly educational for everybody involved. This is particularly appropriate if it is a feature you would like to use, and so can respond critically about whether the PR meets your needs. While each pull request needs to be signed off by two core developers, you can speed up this process by providing your feedback.

Note

The difference between an objective improvement and a subjective nit isn’t always clear. Reviewers should recall that code review is primarily about reducing risk in the project. When reviewing code, one should aim at preventing situations which may require a bug fix, a deprecation, or a retraction. Regarding docs: typos, grammar issues and disambiguations are better addressed immediately.

Important aspects to be covered in any code review

Here are a few important aspects that need to be covered in any code review, from high-level questions to a more detailed check-list.

- Do we want this in the library? Is it likely to be used? Do you, as a scikit-learn user, like the change and intend to use it? Is it in the scope of scikit-learn? Will the cost of maintaining a new feature be worth its benefits?

- Is the code consistent with the API of scikit-learn? Are public functions/classes/parameters well named and intuitively designed?

- Are all public functions/classes and their parameters, return types, and stored attributes named according to scikit-learn conventions and documented clearly?

- Is any new functionality described in the user-guide and illustrated with examples?

- Is every public function/class tested? Are a reasonable set of parameters, their values, value types, and combinations tested? Do the tests validate that the code is correct, i.e. doing what the documentation says it does? If the change is a bug-fix, is a non-regression test included? These tests verify the correct behavior of the fix or feature. In this manner, further modifications on the code base are granted to be consistent with the desired behavior. In the case of bug fixes, at the time of the PR, the non-regression tests should fail for the code base in the `main` branch and pass for the PR code.

- Do the tests pass in the continuous integration build? If appropriate, help the contributor understand why tests failed.

- Do the tests cover every line of code (see the coverage report in the build log)? If not, are the lines missing coverage good exceptions?

- Is the code easy to read and low on redundancy? Should variable names be improved for clarity or consistency? Should comments be added? Should comments be removed as unhelpful or extraneous?

- Could the code easily be rewritten to run much more efficiently for relevant settings?

- Is the code backwards compatible with previous versions? (or is a deprecation cycle necessary?)

- Will the new code add any dependencies on other libraries? (this is unlikely to be accepted)

- Does the documentation render properly (see the [Documentation](https://scikit-learn.org/stable/developers/contributing.html#contribute-documentation) section for more details), and are the plots instructive?

[Standard replies for reviewing](https://scikit-learn.org/stable/developers/tips.html#saved-replies) includes some frequent comments that reviewers may make.

Communication Guidelines

Reviewing open pull requests (PRs) helps move the project forward. It is a great way to get familiar with the codebase and should motivate the contributor to keep involved in the project. [[1]](https://scikit-learn.org/stable/developers/contributing.html#id13)

- Every PR, good or bad, is an act of generosity. Opening with a positive comment will help the author feel rewarded, and your subsequent remarks may be heard more clearly. You may feel good also.

- Begin if possible with the large issues, so the author knows they’ve been understood. Resist the temptation to immediately go line by line, or to open with small pervasive issues.

- Do not let perfect be the enemy of the good. If you find yourself making many small suggestions that don’t fall into the [Code Review Guidelines](https://scikit-learn.org/stable/developers/contributing.html#code-review), consider the following approaches:

  - refrain from submitting these;

  - prefix them as “Nit” so that the contributor knows it’s OK not to address;

  - follow up in a subsequent PR, out of courtesy, you may want to let the original contributor know.

- Do not rush, take the time to make your comments clear and justify your suggestions.

- You are the face of the project. Bad days occur to everyone, in that occasion you deserve a break: try to take your time and stay offline.

\[[1](https://scikit-learn.org/stable/developers/contributing.html#id12)\]

Adapted from the numpy [communication guidelines](https://numpy.org/devdocs/dev/reviewer_guidelines.html#communication-guidelines).

### Reading the existing code base

Reading and digesting an existing code base is always a difficult exercise that takes time and experience to master. Even though we try to write simple code in general, understanding the code can seem overwhelming at first, given the sheer size of the project. Here is a list of tips that may help make this task easier and faster (in no particular order).

- Get acquainted with the [APIs of scikit-learn objects](https://scikit-learn.org/stable/developers/develop.html#api-overview): understand what [fit](https://scikit-learn.org/stable/glossary.html#term-fit), [predict](https://scikit-learn.org/stable/glossary.html#term-predict), [transform](https://scikit-learn.org/stable/glossary.html#term-transform), etc. are used for.

- Before diving into reading the code of a function / class, go through the docstrings first and try to get an idea of what each parameter / attribute is doing. It may also help to stop a minute and think *how would I do this myself if I had to?*

- The trickiest thing is often to identify which portions of the code are relevant, and which are not. In scikit-learn **a lot** of input checking is performed, especially at the beginning of the [fit](https://scikit-learn.org/stable/glossary.html#term-fit) methods. Sometimes, only a very small portion of the code is doing the actual job. For example looking at the [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit "sklearn.linear_model.LinearRegression.fit") method of [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression"), what you’re looking for might just be the call the [`scipy.linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq "(in SciPy v1.17.0)"), but it is buried into multiple lines of input checking and the handling of different kinds of parameters.

- Due to the use of [Inheritance](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>), some methods may be implemented in parent classes. All estimators inherit at least from [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator "sklearn.base.BaseEstimator"), and from a `Mixin` class (e.g. [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin "sklearn.base.ClassifierMixin")) that enables default behaviour depending on the nature of the estimator (classifier, regressor, transformer, etc.).

- Sometimes, reading the tests for a given function will give you an idea of what its intended purpose is. You can use `git grep` (see below) to find all the tests written for a function. Most tests for a specific function/class are placed under the `tests/` folder of the module

- You’ll often see code looking like this: `out = Parallel(...)(delayed(some_function)(param) for param in some_iterable)`. This runs `some_function` in parallel using [Joblib](https://joblib.readthedocs.io/). `out` is then an iterable containing the values returned by `some_function` for each call.

- We use [Cython](https://cython.org/) to write fast code. Cython code is located in `.pyx` and `.pxd` files. Cython code has a more C-like flavor: we use pointers, perform manual memory allocation, etc. Having some minimal experience in C / C++ is pretty much mandatory here. For more information see [Cython Best Practices, Conventions and Knowledge](https://scikit-learn.org/stable/developers/cython.html#cython).

- Master your tools.

  - With such a big project, being efficient with your favorite editor or IDE goes a long way towards digesting the code base. Being able to quickly jump (or *peek*) to a function/class/attribute definition helps a lot. So does being able to quickly see where a given name is used in a file.

  - [Git](https://git-scm.com/book/en) also has some built-in killer features. It is often useful to understand how a file changed over time, using e.g. `git blame` ([manual](https://git-scm.com/docs/git-blame)). This can also be done directly on GitHub. `git grep` ([examples](https://git-scm.com/docs/git-grep#_examples)) is also extremely useful to see every occurrence of a pattern (e.g. a function call or a variable) in the code base.

- Configure `git blame` to ignore the commit that migrated the code style to `black` and then `ruff`.

  ```
  git config blame.ignoreRevsFile .git-blame-ignore-revs
  ```

  Find out more information in black’s [documentation for avoiding ruining git blame](https://black.readthedocs.io/en/stable/guides/introducing_black_to_your_project.html#avoiding-ruining-git-blame).

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/development_setup.html>

## Set up your development environment

### Fork the scikit-learn repository

First, you need to [create an account](https://github.com/join) on GitHub (if you do not already have one) and fork the [project repository](https://github.com/scikit-learn/scikit-learn) by clicking on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

The following steps explain how to set up a local clone of your forked git repository and how to locally install scikit-learn according to your operating system.

### Set up a local clone of your fork

Clone your fork of the scikit-learn repo from your GitHub account to your local disk:

```
git clone https://github.com/YourLogin/scikit-learn.git  # add --depth 1 if your connection is slow
```

and change into that directory:

```
cd scikit-learn
```

Next, add the `upstream` remote. This saves a reference to the main scikit-learn repository, which you can use to keep your repository synchronized with the latest changes (you’ll need this later in the [Development workflow](https://scikit-learn.org/stable/developers/contributing.html#development-workflow)):

```
git remote add upstream https://github.com/scikit-learn/scikit-learn.git
```

Check that the `upstream` and `origin` remote aliases are configured correctly by running:

```
git remote -v
```

This should display:

```
origin    https://github.com/YourLogin/scikit-learn.git (fetch)
origin    https://github.com/YourLogin/scikit-learn.git (push)
upstream  https://github.com/scikit-learn/scikit-learn.git (fetch)
upstream  https://github.com/scikit-learn/scikit-learn.git (push)
```

### Set up a dedicated environment and install dependencies

Using an isolated environment such as [venv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) makes it possible to install a specific version of scikit-learn with pip or conda and its dependencies, independently of any previously installed Python packages, which will avoid potential conflicts with other packages.

In addition to the required Python dependencies, you need to have a working C/C++ compiler with [OpenMP](https://en.wikipedia.org/wiki/OpenMP) support to build scikit-learn [cython](https://cython.org) extensions. The platform-specific instructions below describe how to set up a suitable compiler and install the required packages.

Windows

conda

First, you need to install a compiler with [OpenMP](https://en.wikipedia.org/wiki/OpenMP) support. Download the [Build Tools for Visual Studio installer](https://aka.ms/vs/17/release/vs_buildtools.exe) and run the downloaded `vs_buildtools.exe` file. During the installation you will need to make sure you select “Desktop development with C++”, similarly to this screenshot:

![../images/visual-studio-build-tools-selection.png](https://scikit-learn.org/stable/_images/visual-studio-build-tools-selection.png)

Next, Download and install [the conda-forge installer](https://conda-forge.org/download/) (Miniforge) for your system. Conda-forge provides a conda-based distribution of Python and the most popular scientific libraries. Open the downloaded “Miniforge Prompt” and create a new conda environment with the required python packages:

```
conda create -n sklearn-dev -c conda-forge ^
  python numpy scipy narwhals cython meson-python ninja ^
  pytest pytest-cov ruff==0.12.2 mypy numpydoc ^
  joblib threadpoolctl pre-commit
```

Activate the newly created conda environment:

```
conda activate sklearn-dev
```

pip

First, you need to install a compiler with [OpenMP](https://en.wikipedia.org/wiki/OpenMP) support. Download the [Build Tools for Visual Studio installer](https://aka.ms/vs/17/release/vs_buildtools.exe) and run the downloaded `vs_buildtools.exe` file. During the installation you will need to make sure you select “Desktop development with C++”, similarly to this screenshot:

![../images/visual-studio-build-tools-selection.png](https://scikit-learn.org/stable/_images/visual-studio-build-tools-selection.png)

Next, install the 64-bit version of Python (3.11 or later), for instance from the [official website](https://www.python.org/downloads/windows/).

Now create a virtual environment ([venv](https://docs.python.org/3/tutorial/venv.html)) and install the required python packages:

```
python -m venv sklearn-dev

sklearn-dev\Scripts\activate  # activate

pip install wheel numpy scipy cython meson-python ninja ^
  pytest pytest-cov ruff==0.12.2 mypy numpydoc ^
  joblib threadpoolctl pre-commit
```

MacOS

conda

The default C compiler on macOS does not directly support OpenMP. To enable the installation of the `compilers` meta-package from the conda-forge channel, which provides OpenMP-enabled C/C++ compilers based on the LLVM toolchain, you first need to install the macOS command line tools:

```
xcode-select --install
```

Next, download and install [the conda-forge installer](https://conda-forge.org/download/) (Miniforge) for your system. Conda-forge provides a conda-based distribution of Python and the most popular scientific libraries. Create a new conda environment with the required python packages:

```
conda create -n sklearn-dev -c conda-forge python \
  numpy scipy cython meson-python ninja \
  pytest pytest-cov ruff==0.12.2 mypy numpydoc \
  joblib threadpoolctl compilers llvm-openmp pre-commit
```

and activate the newly created conda environment:

```
conda activate sklearn-dev
```

pip

The default C compiler on macOS does not directly support OpenMP, so you first need to enable OpenMP support.

Install the macOS command line tools:

```
xcode-select --install
```

Next, install the LLVM OpenMP library with [Homebrew](https://brew.sh):

```
brew install libomp
```

Install a recent version of Python (3.11 or later) using [Homebrew](https://brew.sh) (`brew install python`) or by manually installing the package from the [official website](https://www.python.org/downloads/macos/).

Now create a virtual environment ([venv](https://docs.python.org/3/tutorial/venv.html)) and install the required python packages:

```
python -m venv sklearn-dev

source sklearn-dev/bin/activate  # activate

pip install wheel numpy scipy cython meson-python ninja \
  pytest pytest-cov ruff==0.12.2 mypy numpydoc \
  joblib threadpoolctl pre-commit
```

Linux

conda

Download and install [the conda-forge installer](https://conda-forge.org/download/) (Miniforge) for your system. Conda-forge provides a conda-based distribution of Python and the most popular scientific libraries. Create a new conda environment with the required python packages (including `compilers` for a working C/C++ compiler with OpenMP support):

```
conda create -n sklearn-dev -c conda-forge python \
  numpy scipy cython meson-python ninja \
  pytest pytest-cov ruff==0.12.2 mypy numpydoc \
  joblib threadpoolctl compilers pre-commit
```

and activate the newly created environment:

```
conda activate sklearn-dev
```

pip

To check your installed Python version, run:

```
python3 --version
```

If you don’t have Python 3.11 or later, please install `python3` from your distribution’s package manager.

Next, you need to install the build dependencies, specifically a C/C++ compiler with OpenMP support for your system. Here you find the commands for the most widely used distributions:

- On debian-based distributions (e.g., Ubuntu), the compiler is included in the `build-essential` package, and you also need the Python header files:

  ```
  sudo apt-get install build-essential python3-dev
  ```

- On redhat-based distributions (e.g. CentOS), install `` gcc` `` for C and C++, as well as the Python header files:

  ```
  sudo yum -y install gcc gcc-c++ python3-devel
  ```

- On Arche Linux, the Python header files are already included in the python installation, and `` gcc` `` includes the required compilers for C and C++:

  ```
  sudo pacman -S gcc
  ```

Now create a virtual environment ([venv](https://docs.python.org/3/tutorial/venv.html)) and install the required python packages:

```
python -m venv sklearn-dev

source sklearn-dev/bin/activate  # activate

pip install wheel numpy scipy cython meson-python ninja \
  pytest pytest-cov ruff==0.12.2 mypy numpydoc \
  joblib threadpoolctl pre-commit
```

### Install editable version of scikit-learn

Make sure you are in the `scikit-learn` directory and your venv or conda `sklearn-dev` environment is activated. You can now install an editable version of scikit-learn with `pip`:

```
pip install --editable . --verbose --no-build-isolation --config-settings editable-verbose=true
```

Note on `--config-settings`

`--config-settings editable-verbose=true` is optional but recommended to avoid surprises when you import `sklearn`. `meson-python` implements editable installs by rebuilding `sklearn` when executing `import sklearn`. With the recommended setting you will see a message when this happens, rather than potentially waiting without feedback and wondering what is taking so long. Bonus: this means you only have to run the `pip install` command once, `sklearn` will automatically be rebuilt when importing `sklearn`.

Note that `--config-settings` is only supported in `pip` version 23.1 or later. To upgrade `pip` to a compatible version, run `pip install -U pip`.

To check your installation, make sure that the installed scikit-learn has a version number ending with `.dev0`:

```
python -c "import sklearn; sklearn.show_versions()"
```

You should now have a working installation of scikit-learn and your git repository properly configured.

It can be useful to run the tests now (even though it will take some time) to verify your installation and to be aware of warnings and errors that are not related to you contribution:

```
pytest
```

For more information on testing, see also the [Pull request checklist](https://scikit-learn.org/stable/developers/contributing.html#pr-checklist) and [Useful pytest aliases and flags](https://scikit-learn.org/stable/developers/tips.html#pytest-tips).

### Set up pre-commit

Additionally, install the [pre-commit hooks](https://pre-commit.com), which will automatically check your code for linting problems before each commit in the [Development workflow](https://scikit-learn.org/stable/developers/contributing.html#development-workflow):

```
pre-commit install
```

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/minimal_reproducer.html>

## Crafting a minimal reproducer for scikit-learn

Whether submitting a bug report, designing a suite of tests, or simply posting a question in the discussions, being able to craft minimal, reproducible examples (or minimal, workable examples) is the key to communicating effectively and efficiently with the community.

There are very good guidelines on the internet such as [this StackOverflow document](https://stackoverflow.com/help/mcve) or [this blogpost by Matthew Rocklin](https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports) on crafting Minimal Complete Verifiable Examples (referred below as MCVE). Our goal is not to be repetitive with those references but rather to provide a step-by-step guide on how to narrow down a bug until you have reached the shortest possible code to reproduce it.

The first step before submitting a bug report to scikit-learn is to read the [Issue template](https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml). It is already quite informative about the information you will be asked to provide.

### Good practices

In this section we will focus on the **Steps/Code to Reproduce** section of the [Issue template](https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml). We will start with a snippet of code that already provides a failing example but that has room for readability improvement. We then craft an MCVE from it.

**Example**

```
# I am currently working in an ML project and when I tried to fit a
# GradientBoostingRegressor instance to my_data.csv I get a UserWarning:
# "X has feature names, but DecisionTreeRegressor was fitted without
# feature names". You can get a copy of my dataset from
# https://example.com/my_data.csv and verify my features do have
# names. The problem seems to arise during fit when I pass an integer
# to the n_iter_no_change parameter.

df = pd.read_csv('my_data.csv')
X = df[["feature_name"]] # my features do have names
y = df["target"]

# We set random_state=42 for the train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# An instance with default n_iter_no_change raises no error nor warnings
gbdt = GradientBoostingRegressor(random_state=0)
gbdt.fit(X_train, y_train)
default_score = gbdt.score(X_test, y_test)

# the bug appears when I change the value for n_iter_no_change
gbdt = GradientBoostingRegressor(random_state=0, n_iter_no_change=5)
gbdt.fit(X_train, y_train)
other_score = gbdt.score(X_test, y_test)

other_score = gbdt.score(X_test, y_test)
```

#### Provide a failing code example with minimal comments

Writing instructions to reproduce the problem in English is often ambiguous. Better make sure that all the necessary details to reproduce the problem are illustrated in the Python code snippet to avoid any ambiguity. Besides, by this point you already provided a concise description in the **Describe the bug** section of the [Issue template](https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml).

The following code, while **still not minimal**, is already **much better** because it can be copy-pasted in a Python terminal to reproduce the problem in one step. In particular:

- it contains **all necessary import statements**;

- it can fetch the public dataset without having to manually download a file and put it in the expected location on the disk.

**Improved example**

```
import pandas as pd

df = pd.read_csv("https://example.com/my_data.csv")
X = df[["feature_name"]]
y = df["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingRegressor

gbdt = GradientBoostingRegressor(random_state=0)
gbdt.fit(X_train, y_train)  # no warning
default_score = gbdt.score(X_test, y_test)

gbdt = GradientBoostingRegressor(random_state=0, n_iter_no_change=5)
gbdt.fit(X_train, y_train)  # raises warning
other_score = gbdt.score(X_test, y_test)
other_score = gbdt.score(X_test, y_test)
```

#### Boil down your script to something as small as possible

You have to ask yourself which lines of code are relevant and which are not for reproducing the bug. Deleting unnecessary lines of code or simplifying the function calls by omitting unrelated non-default options will help you and other contributors narrow down the cause of the bug.

In particular, for this specific example:

- the warning has nothing to do with the `train_test_split` since it already appears in the training step, before we use the test set.

- similarly, the lines that compute the scores on the test set are not necessary;

- the bug can be reproduced for any value of `random_state` so leave it to its default;

- the bug can be reproduced without preprocessing the data with the `StandardScaler`.

**Improved example**

```
import pandas as pd
df = pd.read_csv("https://example.com/my_data.csv")
X = df[["feature_name"]]
y = df["target"]

from sklearn.ensemble import GradientBoostingRegressor

gbdt = GradientBoostingRegressor()
gbdt.fit(X, y)  # no warning

gbdt = GradientBoostingRegressor(n_iter_no_change=5)
gbdt.fit(X, y)  # raises warning
```

#### **DO NOT** report your data unless it is extremely necessary

The idea is to make the code as self-contained as possible. For doing so, you can use a [Synthetic dataset](https://scikit-learn.org/stable/developers/minimal_reproducer.html#synth-data). It can be generated using numpy, pandas or the [`sklearn.datasets`](https://scikit-learn.org/stable/api/sklearn.datasets.html#module-sklearn.datasets "sklearn.datasets") module. Most of the times the bug is not related to a particular structure of your data. Even if it is, try to find an available dataset that has similar characteristics to yours and that reproduces the problem. In this particular case, we are interested in data that has labeled feature names.

**Improved example**

```
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.DataFrame(
    {
        "feature_name": [-12.32, 1.43, 30.01, 22.17],
        "target": [72, 55, 32, 43],
    }
)
X = df[["feature_name"]]
y = df["target"]

gbdt = GradientBoostingRegressor()
gbdt.fit(X, y) # no warning
gbdt = GradientBoostingRegressor(n_iter_no_change=5)
gbdt.fit(X, y) # raises warning
```

As already mentioned, the key to communication is the readability of the code and good formatting can really be a plus. Notice that in the previous snippet we:

- try to limit all lines to a maximum of 79 characters to avoid horizontal scrollbars in the code snippets blocks rendered on the GitHub issue;

- use blank lines to separate groups of related functions;

- place all the imports in their own group at the beginning.

The simplification steps presented in this guide can be implemented in a different order than the progression we have shown here. The important points are:

- a minimal reproducer should be runnable by a simple copy-and-paste in a python terminal;

- it should be simplified as much as possible by removing any code steps that are not strictly needed to reproducing the original problem;

- it should ideally only rely on a minimal dataset generated on-the-fly by running the code instead of relying on external data, if possible.

#### Use markdown formatting

To format code or text into its own distinct block, use triple backticks. [Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) supports an optional language identifier to enable syntax highlighting in your fenced code block. For example:

````
```python
from sklearn.datasets import make_blobs

n_samples = 100
n_components = 3
X, y = make_blobs(n_samples=n_samples, centers=n_components)
```
````

will render a python formatted snippet as follows

```
from sklearn.datasets import make_blobs

n_samples = 100
n_components = 3
X, y = make_blobs(n_samples=n_samples, centers=n_components)
```

It is not necessary to create several blocks of code when submitting a bug report. Remember other reviewers are going to copy-paste your code and having a single cell will make their task easier.

In the section named **Actual results** of the [Issue template](https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml) you are asked to provide the error message including the full traceback of the exception. In this case, use the `python-traceback` qualifier. For example:

````
```python-traceback
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-1-a674e682c281> in <module>
    4 vectorizer = CountVectorizer(input=docs, analyzer='word')
    5 lda_features = vectorizer.fit_transform(docs)
----> 6 lda_model = LatentDirichletAllocation(
    7     n_topics=10,
    8     learning_method='online',

TypeError: __init__() got an unexpected keyword argument 'n_topics'
```
````

yields the following when rendered:

```
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-1-a674e682c281> in <module>
    4 vectorizer = CountVectorizer(input=docs, analyzer='word')
    5 lda_features = vectorizer.fit_transform(docs)
----> 6 lda_model = LatentDirichletAllocation(
    7     n_topics=10,
    8     learning_method='online',

TypeError: __init__() got an unexpected keyword argument 'n_topics'
```

### Synthetic dataset

Before choosing a particular synthetic dataset, first you have to identify the type of problem you are solving: Is it a classification, a regression, a clustering, etc?

Once that you narrowed down the type of problem, you need to provide a synthetic dataset accordingly. Most of the times you only need a minimalistic dataset. Here is a non-exhaustive list of tools that may help you.

#### NumPy

NumPy tools such as [numpy.random.randn](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html) and [numpy.random.randint](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html) can be used to create dummy numeric data.

- regression

  Regressions take continuous numeric data as features and target.

  ```
  import numpy as np

  rng = np.random.RandomState(0)
  n_samples, n_features = 5, 5
  X = rng.randn(n_samples, n_features)
  y = rng.randn(n_samples)
  ```

A similar snippet can be used as synthetic data when testing scaling tools such as [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler "sklearn.preprocessing.StandardScaler").

- classification

  If the bug is not raised during when encoding a categorical variable, you can feed numeric data to a classifier. Just remember to ensure that the target is indeed an integer.

  ```
  import numpy as np

  rng = np.random.RandomState(0)
  n_samples, n_features = 5, 5
  X = rng.randn(n_samples, n_features)
  y = rng.randint(0, 2, n_samples)  # binary target with values in {0, 1}
  ```

  If the bug only happens with non-numeric class labels, you might want to generate a random target with [numpy.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html).

  ```
  import numpy as np

  rng = np.random.RandomState(0)
  n_samples, n_features = 50, 5
  X = rng.randn(n_samples, n_features)
  y = np.random.choice(
      ["male", "female", "other"], size=n_samples, p=[0.49, 0.49, 0.02]
  )
  ```

#### Pandas

Some scikit-learn objects expect pandas dataframes as input. In this case you can transform numpy arrays into pandas objects using [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), or [pandas.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html).

```
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
n_samples, n_features = 5, 5
X = pd.DataFrame(
    {
        "continuous_feature": rng.randn(n_samples),
        "positive_feature": rng.uniform(low=0.0, high=100.0, size=n_samples),
        "categorical_feature": rng.choice(["a", "b", "c"], size=n_samples),
    }
)
y = pd.Series(rng.randn(n_samples))
```

In addition, scikit-learn includes various [Generated datasets](https://scikit-learn.org/stable/datasets/sample_generators.html#sample-generators) that can be used to build artificial datasets of controlled size and complexity.

#### `make_regression`

As hinted by the name, [`sklearn.datasets.make_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression "sklearn.datasets.make_regression") produces regression targets with noise as an optionally-sparse random linear combination of random features.

```
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=20)
```

#### `make_classification`

[`sklearn.datasets.make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification "sklearn.datasets.make_classification") creates multiclass datasets with multiple Gaussian clusters per class. Noise can be introduced by means of correlated, redundant or uninformative features.

```
from sklearn.datasets import make_classification

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
)
```

#### `make_blobs`

Similarly to `make_classification`, [`sklearn.datasets.make_blobs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs "sklearn.datasets.make_blobs") creates multiclass datasets using normally-distributed clusters of points. It provides greater control regarding the centers and standard deviations of each cluster, and therefore it is useful to demonstrate clustering.

```
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=10, centers=3, n_features=2)
```

#### Dataset loading utilities

You can use the [Dataset loading utilities](https://scikit-learn.org/stable/datasets.html#datasets) to load and fetch several popular reference datasets. This option is useful when the bug relates to the particular structure of the data, e.g. dealing with missing values or image recognition.

```
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
```

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/develop.html>

## Developing scikit-learn estimators

Whether you are proposing an estimator for inclusion in scikit-learn, developing a separate package compatible with scikit-learn, or implementing custom components for your own projects, this chapter details how to develop objects that safely interact with scikit-learn pipelines and model selection tools.

This section details the public API you should use and implement for a scikit-learn compatible estimator. Inside scikit-learn itself, we experiment and use some private tools and our goal is always to make them public once they are stable enough, so that you can also use them in your own projects.

### APIs of scikit-learn objects

There are two major types of estimators. You can think of the first group as simple estimators, which consists of most estimators, such as [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") or [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier "sklearn.ensemble.RandomForestClassifier"). And the second group are meta-estimators, which are estimators that wrap other estimators. [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline") and [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV") are two examples of meta-estimators.

Here we start with a few vocabulary terms, and then we illustrate how you can implement your own estimators.

Elements of the scikit-learn API are described more definitively in the [Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html#glossary).

#### Different objects

The main objects in scikit-learn are (one class can implement multiple interfaces):

Estimator:
The base object, implements a `fit` method to learn from data, either:

```
estimator = estimator.fit(data, targets)
```

or:

```
estimator = estimator.fit(data)
```

Predictor:
For supervised learning, or some unsupervised problems, implements:

```
prediction = predictor.predict(data)
```

Classification algorithms usually also offer a way to quantify certainty of a prediction, either using `decision_function` or `predict_proba`:

```
probability = predictor.predict_proba(data)
```

Transformer:
For modifying the data in a supervised or unsupervised way (e.g. by adding, changing, or removing columns, but not by adding or removing rows). Implements:

```
new_data = transformer.transform(data)
```

When fitting and transforming can be performed much more efficiently together than separately, implements:

```
new_data = transformer.fit_transform(data)
```

Model:
A model that can give a [goodness of fit](https://en.wikipedia.org/wiki/Goodness_of_fit) measure or a likelihood of unseen data, implements (higher is better):

```
score = model.score(data)
```

#### Estimators

The API has one predominant object: the estimator. An estimator is an object that fits a model based on some training data and is capable of inferring some properties on new data. It can be, for instance, a classifier or a regressor. All estimators implement the fit method:

```
estimator.fit(X, y)
```

Out of all the methods that an estimator implements, `fit` is usually the one you want to implement yourself. Other methods such as `set_params`, `get_params`, etc. are implemented in [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator "sklearn.base.BaseEstimator"), which you should inherit from. You might need to inherit from more mixins, which we will explain later.

##### Instantiation

This concerns the creation of an object. The object’s `__init__` method might accept constants as arguments that determine the estimator’s behavior (like the `alpha` constant in [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier")). It should not, however, take the actual training data as an argument, as this is left to the `fit()` method:

```
clf2 = SGDClassifier(alpha=2.3)
clf3 = SGDClassifier([[1, 2], [2, 3]], [-1, 1]) # WRONG!
```

Ideally, the arguments accepted by `__init__` should all be keyword arguments with a default value. In other words, a user should be able to instantiate an estimator without passing any arguments to it. In some cases, where there are no sane defaults for an argument, they can be left without a default value. In scikit-learn itself, we have very few places, only in some meta-estimators, where the sub-estimator(s) argument is a required argument.

Most arguments correspond to hyperparameters describing the model or the optimisation problem the estimator tries to solve. Other parameters might define how the estimator behaves, e.g. defining the location of a cache to store some data. These initial arguments (or parameters) are always remembered by the estimator. Also note that they should not be documented under the “Attributes” section, but rather under the “Parameters” section for that estimator.

In addition, **every keyword argument accepted by** `__init__` **should correspond to an attribute on the instance**. Scikit-learn relies on this to find the relevant attributes to set on an estimator when doing model selection.

To summarize, an `__init__` should look like:

```
def __init__(self, param1=1, param2=2):
    self.param1 = param1
    self.param2 = param2
```

There should be no logic, not even input validation, and the parameters should not be changed; which also means ideally they should not be mutable objects such as lists or dictionaries. If they’re mutable, they should be copied before being modified. The corresponding logic should be put where the parameters are used, typically in `fit`. The following is wrong:

```
def __init__(self, param1=1, param2=2, param3=3):
    # WRONG: parameters should not be modified
    if param1 > 1:
        param2 += 1
    self.param1 = param1
    # WRONG: the object's attributes should have exactly the name of
    # the argument in the constructor
    self.param3 = param2
```

The reason for postponing the validation is that if `__init__` includes input validation, then the same validation would have to be performed in `set_params`, which is used in algorithms like [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV").

Also it is expected that parameters with trailing `_` are **not to be set inside the** `__init__` **method**. More details on attributes that are not init arguments come shortly.

##### Fitting

The next thing you will probably want to do is to estimate some parameters in the model. This is implemented in the `fit()` method, and it’s where the training happens. For instance, this is where you have the computation to learn or estimate coefficients for a linear model.

The `fit()` method takes the training data as arguments, which can be one array in the case of unsupervised learning, or two arrays in the case of supervised learning. Other metadata that come with the training data, such as `sample_weight`, can also be passed to `fit` as keyword arguments.

Note that the model is fitted using `X` and `y`, but the object holds no reference to `X` and `y`. There are, however, some exceptions to this, as in the case of precomputed kernels where this data must be stored for use by the predict method.

| Parameters |                                             |
| ---------- | ------------------------------------------- |
| X          | array-like of shape (n_samples, n_features) |
| y          | array-like of shape (n_samples,)            |
| kwargs     | optional data-dependent parameters          |

The number of samples, i.e. `X.shape[0]` should be the same as `y.shape[0]`. If this requirement is not met, an exception of type `ValueError` should be raised.

`y` might be ignored in the case of unsupervised learning. However, to make it possible to use the estimator as part of a pipeline that can mix both supervised and unsupervised transformers, even unsupervised estimators need to accept a `y=None` keyword argument in the second position that is just ignored by the estimator. For the same reason, `fit_predict`, `fit_transform`, `score` and `partial_fit` methods need to accept a `y` argument in the second place if they are implemented.

The method should return the object (`self`). This pattern is useful to be able to implement quick one liners in an IPython session such as:

```
y_predicted = SGDClassifier(alpha=10).fit(X_train, y_train).predict(X_test)
```

Depending on the nature of the algorithm, `fit` can sometimes also accept additional keywords arguments. However, any parameter that can have a value assigned prior to having access to the data should be an `__init__` keyword argument. Ideally, **fit parameters should be restricted to directly data dependent variables**. For instance a Gram matrix or an affinity matrix which are precomputed from the data matrix `X` are data dependent. A tolerance stopping criterion `tol` is not directly data dependent (although the optimal value according to some scoring function probably is).

When `fit` is called, any previous call to `fit` should be ignored. In general, calling `estimator.fit(X1)` and then `estimator.fit(X2)` should be the same as only calling `estimator.fit(X2)`. However, this may not be true in practice when `fit` depends on some random process, see [random_state](https://scikit-learn.org/stable/glossary.html#term-random_state). Another exception to this rule is when the hyper-parameter `warm_start` is set to `True` for estimators that support it. `warm_start=True` means that the previous state of the trainable parameters of the estimator are reused instead of using the default initialization strategy.

##### Estimated Attributes

According to scikit-learn conventions, attributes which you’d want to expose to your users as public attributes and have been estimated or learned from the data must always have a name ending with trailing underscore, for example the coefficients of some regression estimator would be stored in a `coef_` attribute after `fit` has been called. Similarly, attributes that you learn in the process and you’d like to store yet not expose to the user, should have a leading underscore, e.g. `_intermediate_coefs`. You’d need to document the first group (with a trailing underscore) as “Attributes” and no need to document the second group (with a leading underscore).

The estimated attributes are expected to be overridden when you call `fit` a second time.

##### Universal attributes

Estimators that expect tabular input should set a `n_features_in_` attribute at `fit` time to indicate the number of features that the estimator expects for subsequent calls to [predict](https://scikit-learn.org/stable/glossary.html#term-predict) or [transform](https://scikit-learn.org/stable/glossary.html#term-transform). See [SLEP010](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html) for details.

Similarly, if estimators are given dataframes such as pandas or polars, they should set a `feature_names_in_` attribute to indicate the features names of the input data, detailed in [SLEP007](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html). Using [`validate_data`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.validate_data.html#sklearn.utils.validation.validate_data "sklearn.utils.validation.validate_data") would automatically set these attributes for you.

### Rolling your own estimator

If you want to implement a new estimator that is scikit-learn compatible, there are several internals of scikit-learn that you should be aware of in addition to the scikit-learn API outlined above. You can check whether your estimator adheres to the scikit-learn interface and standards by running [`check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator "sklearn.utils.estimator_checks.check_estimator") on an instance. The [`parametrize_with_checks`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks "sklearn.utils.estimator_checks.parametrize_with_checks") pytest decorator can also be used (see its docstring for details and possible interactions with `pytest`):

```
>>> from sklearn.utils.estimator_checks import check_estimator
>>> from sklearn.tree import DecisionTreeClassifier
>>> check_estimator(DecisionTreeClassifier())  # passes
[...]
```

The main motivation to make a class compatible to the scikit-learn estimator interface might be that you want to use it together with model evaluation and selection tools such as [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV") and [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline").

Before detailing the required interface below, we describe two ways to achieve the correct interface more easily.

Project template:

We provide a [project template](https://github.com/scikit-learn-contrib/project-template/) which helps in the creation of Python packages containing scikit-learn compatible estimators. It provides:

- an initial git repository with Python package directory structure

- a template of a scikit-learn estimator

- an initial test suite including use of `parametrize_with_checks`

- directory structures and scripts to compile documentation and example galleries

- scripts to manage continuous integration (testing on Linux, MacOS, and Windows)

- instructions from getting started to publishing on [PyPi](https://pypi.org/)

[`base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator "sklearn.base.BaseEstimator") and mixins:

We tend to use “duck typing” instead of checking for [`isinstance`](https://docs.python.org/3/library/functions.html#isinstance "(in Python v3.14)"), which means it’s technically possible to implement an estimator without inheriting from scikit-learn classes. However, if you don’t inherit from the right mixins, either there will be a large amount of boilerplate code for you to implement and keep in sync with scikit-learn development, or your estimator might not function the same way as a scikit-learn estimator. Here we only document how to develop an estimator using our mixins. If you’re interested in implementing your estimator without inheriting from scikit-learn mixins, you’d need to check our implementations.

For example, below is a custom classifier, with more examples included in the scikit-learn-contrib [project template](https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py).

It is particularly important to notice that mixins should be “on the left” while the `BaseEstimator` should be “on the right” in the inheritance list for proper MRO.

```
>>> import numpy as np
>>> from sklearn.base import BaseEstimator, ClassifierMixin
>>> from sklearn.utils.validation import validate_data, check_is_fitted
>>> from sklearn.utils.multiclass import unique_labels
>>> from sklearn.metrics import euclidean_distances
>>> class TemplateClassifier(ClassifierMixin, BaseEstimator):
...
...     def __init__(self, demo_param='demo'):
...         self.demo_param = demo_param
...
...     def fit(self, X, y):
...
...         # Check that X and y have correct shape, set n_features_in_, etc.
...         X, y = validate_data(self, X, y)
...         # Store the classes seen during fit
...         self.classes_ = unique_labels(y)
...
...         self.X_ = X
...         self.y_ = y
...         # Return the classifier
...         return self
...
...     def predict(self, X):
...
...         # Check if fit has been called
...         check_is_fitted(self)
...
...         # Input validation
...         X = validate_data(self, X, reset=False)
...
...         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
...         return self.y_[closest]
```

And you can check that the above estimator passes all common checks:

```
>>> from sklearn.utils.estimator_checks import check_estimator
>>> check_estimator(TemplateClassifier())  # passes
```

#### get_params and set_params

All scikit-learn estimators have `get_params` and `set_params` functions.

The `get_params` function takes no arguments and returns a dict of the `__init__` parameters of the estimator, together with their values.

It takes one keyword argument, `deep`, which receives a boolean value that determines whether the method should return the parameters of sub-estimators (only relevant for meta-estimators). The default value for `deep` is `True`. For instance considering the following estimator:

```
>>> from sklearn.base import BaseEstimator
>>> from sklearn.linear_model import LogisticRegression
>>> class MyEstimator(BaseEstimator):
...     def __init__(self, subestimator=None, my_extra_param="random"):
...         self.subestimator = subestimator
...         self.my_extra_param = my_extra_param
```

The parameter `deep` controls whether or not the parameters of the `subestimator` should be reported. Thus when `deep=True`, the output will be:

```
>>> my_estimator = MyEstimator(subestimator=LogisticRegression())
>>> for param, value in my_estimator.get_params(deep=True).items():
...     print(f"{param} -> {value}")
my_extra_param -> random
subestimator__C -> 1.0
subestimator__class_weight -> None
subestimator__dual -> False
subestimator__fit_intercept -> True
subestimator__intercept_scaling -> 1
subestimator__l1_ratio -> 0.0
subestimator__max_iter -> 100
subestimator__n_jobs -> None
subestimator__penalty -> deprecated
subestimator__random_state -> None
subestimator__solver -> lbfgs
subestimator__tol -> 0.0001
subestimator__verbose -> 0
subestimator__warm_start -> False
subestimator -> LogisticRegression()
```

If the meta-estimator takes multiple sub-estimators, often, those sub-estimators have names (as e.g. named steps in a [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline") object), in which case the key should become `<name>__C`, `<name>__class_weight`, etc.

When `deep=False`, the output will be:

```
>>> for param, value in my_estimator.get_params(deep=False).items():
...     print(f"{param} -> {value}")
my_extra_param -> random
subestimator -> LogisticRegression()
```

On the other hand, `set_params` takes the parameters of `__init__` as keyword arguments, unpacks them into a dict of the form `'parameter': value` and sets the parameters of the estimator using this dict. It returns the estimator itself.

The [`set_params`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_params "sklearn.base.BaseEstimator.set_params") function is used to set parameters during grid search for instance.

#### Cloning

As already mentioned that when constructor arguments are mutable, they should be copied before modifying them. This also applies to constructor arguments which are estimators. That’s why meta-estimators such as [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV") create a copy of the given estimator before modifying it.

However, in scikit-learn, when we copy an estimator, we get an unfitted estimator where only the constructor arguments are copied (with some exceptions, e.g. attributes related to certain internal machinery such as metadata routing).

The function responsible for this behavior is [`clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone "sklearn.base.clone").

Estimators can customize the behavior of [`base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone "sklearn.base.clone") by overriding the `base.BaseEstimator.__sklearn_clone__` method. `__sklearn_clone__` must return an instance of the estimator. `__sklearn_clone__` is useful when an estimator needs to hold on to some state when [`base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone "sklearn.base.clone") is called on the estimator. For example, [`FrozenEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html#sklearn.frozen.FrozenEstimator "sklearn.frozen.FrozenEstimator") makes use of this.

#### Estimator types

Among simple estimators (as opposed to meta-estimators), the most common types are transformers, classifiers, regressors, and clustering algorithms.

**Transformers** inherit from [`TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin "sklearn.base.TransformerMixin"), and implement a `transform` method. These are estimators which take the input, and transform it in some way. Note that they should never change the number of input samples, and the output of `transform` should correspond to its input samples in the same given order.

**Regressors** inherit from [`RegressorMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin "sklearn.base.RegressorMixin"), and implement a `predict` method. They should accept numerical `y` in their `fit` method. Regressors use [`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score "sklearn.metrics.r2_score") by default in their [`score`](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin.score "sklearn.base.RegressorMixin.score") method.

**Classifiers** inherit from [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin "sklearn.base.ClassifierMixin"). If it applies, classifiers can implement `decision_function` to return raw decision values, based on which `predict` can make its decision. If calculating probabilities is supported, classifiers can also implement `predict_proba` and `predict_log_proba`.

Classifiers should accept `y` (target) arguments to `fit` that are sequences (lists, arrays) of either strings or integers. They should not assume that the class labels are a contiguous range of integers; instead, they should store a list of classes in a `classes_` attribute or property. The order of class labels in this attribute should match the order in which `predict_proba`, `predict_log_proba` and `decision_function` return their values. The easiest way to achieve this is to put:

```
self.classes_, y = np.unique(y, return_inverse=True)
```

in `fit`. This returns a new `y` that contains class indexes, rather than labels, in the range \[0, `n_classes`).

A classifier’s `predict` method should return arrays containing class labels from `classes_`. In a classifier that implements `decision_function`, this can be achieved with:

```
def predict(self, X):
    D = self.decision_function(X)
    return self.classes_[np.argmax(D, axis=1)]
```

The [`multiclass`](https://scikit-learn.org/stable/api/sklearn.utils.html#module-sklearn.utils.multiclass "sklearn.utils.multiclass") module contains useful functions for working with multiclass and multilabel problems.

**Clustering algorithms** inherit from [`ClusterMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin "sklearn.base.ClusterMixin"). Ideally, they should accept a `y` parameter in their `fit` method, but it should be ignored. Clustering algorithms should set a `labels_` attribute, storing the labels assigned to each sample. If applicable, they can also implement a `predict` method, returning the labels assigned to newly given samples.

If one needs to check the type of a given estimator, e.g. in a meta-estimator, one can check if the given object implements a `transform` method for transformers, and otherwise use helper functions such as [`is_classifier`](https://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html#sklearn.base.is_classifier "sklearn.base.is_classifier") or [`is_regressor`](https://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor "sklearn.base.is_regressor").

#### Estimator Tags

Note

Scikit-learn introduced estimator tags in version 0.21 as a private API and mostly used in tests. However, these tags expanded over time and many third party developers also need to use them. Therefore in version 1.6 the API for the tags was revamped and exposed as public API.

The estimator tags are annotations of estimators that allow programmatic inspection of their capabilities, such as sparse matrix support, supported output types and supported methods. The estimator tags are an instance of [`Tags`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags "sklearn.utils.Tags") returned by the method `__sklearn_tags__`. These tags are used in different places, such as [`is_regressor`](https://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor "sklearn.base.is_regressor") or the common checks run by [`check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator "sklearn.utils.estimator_checks.check_estimator") and [`parametrize_with_checks`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks "sklearn.utils.estimator_checks.parametrize_with_checks"), where tags determine which checks to run and what input data is appropriate. Tags can depend on estimator parameters or even system architecture and can in general only be determined at runtime and are therefore instance attributes rather than class attributes. See [`Tags`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags "sklearn.utils.Tags") for more information about individual tags.

It is unlikely that the default values for each tag will suit the needs of your specific estimator. You can change the default values by defining a `__sklearn_tags__()` method which returns the new values for your estimator’s tags. For example:

```
class MyMultiOutputEstimator(BaseEstimator):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.non_deterministic = True
        return tags
```

You can create a new subclass of [`Tags`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags "sklearn.utils.Tags") if you wish to add new tags to the existing set. Note that all attributes that you add in a child class need to have a default value. It can be of the form:

```
from dataclasses import dataclass, fields

@dataclass
class MyTags(Tags):
    my_tag: bool = True

class MyEstimator(BaseEstimator):
    def __sklearn_tags__(self):
        tags_orig = super().__sklearn_tags__()
        as_dict = {
            field.name: getattr(tags_orig, field.name)
            for field in fields(tags_orig)
        }
        tags = MyTags(**as_dict)
        tags.my_tag = True
        return tags
```

### Developer API for `set_output`

With [SLEP018](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep018/proposal.html), scikit-learn introduces the `set_output` API for configuring transformers to output pandas DataFrames. The `set_output` API is automatically defined if the transformer defines [get_feature_names_out](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out) and subclasses [`base.TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin "sklearn.base.TransformerMixin"). [get_feature_names_out](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out) is used to get the column names of pandas output.

[`base.OneToOneFeatureMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.OneToOneFeatureMixin.html#sklearn.base.OneToOneFeatureMixin "sklearn.base.OneToOneFeatureMixin") and [`base.ClassNamePrefixFeaturesOutMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassNamePrefixFeaturesOutMixin.html#sklearn.base.ClassNamePrefixFeaturesOutMixin "sklearn.base.ClassNamePrefixFeaturesOutMixin") are helpful mixins for defining [get_feature_names_out](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out). [`base.OneToOneFeatureMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.OneToOneFeatureMixin.html#sklearn.base.OneToOneFeatureMixin "sklearn.base.OneToOneFeatureMixin") is useful when the transformer has a one-to-one correspondence between input features and output features, such as [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler "sklearn.preprocessing.StandardScaler"). [`base.ClassNamePrefixFeaturesOutMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassNamePrefixFeaturesOutMixin.html#sklearn.base.ClassNamePrefixFeaturesOutMixin "sklearn.base.ClassNamePrefixFeaturesOutMixin") is useful when the transformer needs to generate its own feature names out, such as [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA "sklearn.decomposition.PCA").

You can opt-out of the `set_output` API by setting `auto_wrap_output_keys=None` when defining a custom subclass:

```
class MyTransformer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X
    def get_feature_names_out(self, input_features=None):
        ...
```

The default value for `auto_wrap_output_keys` is `("transform",)`, which automatically wraps `fit_transform` and `transform`. The `TransformerMixin` uses the `__init_subclass__` mechanism to consume `auto_wrap_output_keys` and pass all other keyword arguments to its super class. Super classes’ `__init_subclass__` should **not** depend on `auto_wrap_output_keys`.

For transformers that return multiple arrays in `transform`, auto wrapping will only wrap the first array and not alter the other arrays.

Refer to the [user guide](https://scikit-learn.org/stable/modules/df_output_transform.html#df-output-transform) for more details and [Introducing the set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html#sphx-glr-auto-examples-miscellaneous-plot-set-output-py) for an example on how to use the API.

### Developer API for `check_is_fitted`

By default [`check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted "sklearn.utils.validation.check_is_fitted") checks if there are any attributes in the instance with a trailing underscore, e.g. `coef_`. An estimator can change the behavior by implementing a `__sklearn_is_fitted__` method taking no input and returning a boolean. If this method exists, [`check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted "sklearn.utils.validation.check_is_fitted") simply returns its output.

See [\_\_sklearn_is_fitted\_\_ as Developer API](https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html#sphx-glr-auto-examples-developing-estimators-sklearn-is-fitted-py) for an example on how to use the API.

### Developer API for HTML representation

Warning

The HTML representation API is experimental and the API is subject to change.

Estimators inheriting from [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator "sklearn.base.BaseEstimator") display a HTML representation of themselves in interactive programming environments such as Jupyter notebooks. For instance, we can display this HTML diagram:

```
from sklearn.base import BaseEstimator

BaseEstimator()
```

The raw HTML representation is obtained by invoking the function [`estimator_html_repr`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_html_repr.html#sklearn.utils.estimator_html_repr "sklearn.utils.estimator_html_repr") on an estimator instance.

To customize the URL linking to an estimator’s documentation (i.e. when clicking on the “?” icon), override the `_doc_link_module` and `_doc_link_template` attributes. In addition, you can provide a `_doc_link_url_param_generator` method. Set `_doc_link_module` to the name of the (top level) module that contains your estimator. If the value does not match the top level module name, the HTML representation will not contain a link to the documentation. For scikit-learn estimators this is set to `"sklearn"`.

The `_doc_link_template` is used to construct the final URL. By default, it can contain two variables: `estimator_module` (the full name of the module containing the estimator) and `estimator_name` (the class name of the estimator). If you need more variables you should implement the `_doc_link_url_param_generator` method which should return a dictionary of the variables and their values. This dictionary will be used to render the `_doc_link_template`.

### Coding guidelines

The following are some guidelines on how new code should be written for inclusion in scikit-learn, and which may be appropriate to adopt in external projects. Of course, there are special cases and there will be exceptions to these rules. However, following these rules when submitting new code makes the review easier so new code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The scikit-learn project tries to closely follow the official Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008) that detail how code should be formatted and indented. Please read it and follow it.

In addition, we add the following guidelines:

- Use underscores to separate words in non class names: `n_samples` rather than `nsamples`.

- Avoid multiple statements on one line. Prefer a line return after a control flow statement (`if`/`for`).

- Use absolute imports

- Unit tests should use imports exactly as client code would. If `sklearn.foo` exports a class or function that is implemented in `sklearn.foo.bar.baz`, the test should import it from `sklearn.foo`.

- **Please don’t use** `import *` **in any case**. It is considered harmful by the [official Python recommendations](https://docs.python.org/3.1/howto/doanddont.html#at-module-level). It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like [pyflakes](https://divmod.readthedocs.io/en/latest/products/pyflakes.html) to automatically find bugs in scikit-learn.

- Use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) in all your docstrings.

A good example of code that we like can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01).

#### Input validation

The module [`sklearn.utils`](https://scikit-learn.org/stable/api/sklearn.utils.html#module-sklearn.utils "sklearn.utils") contains various functions for doing input validation and conversion. Sometimes, `np.asarray` suffices for validation; do *not* use `np.asanyarray` or `np.atleast_2d`, since those let NumPy’s `np.matrix` through, which has a different API (e.g., `*` means dot product on `np.matrix`, but Hadamard product on `np.ndarray`).

In other cases, be sure to call [`check_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array "sklearn.utils.check_array") on any array-like argument passed to a scikit-learn API function. The exact parameters to use depends mainly on whether and which `scipy.sparse` matrices must be accepted.

For more information, refer to the [Utilities for Developers](https://scikit-learn.org/stable/developers/utilities.html#developers-utils) page.

#### Random Numbers

If your code depends on a random number generator, do not use `numpy.random.random()` or similar routines. To ensure repeatability in error checking, the routine should accept a keyword `random_state` and use this to construct a `numpy.random.RandomState` object. See [`sklearn.utils.check_random_state`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state "sklearn.utils.check_random_state") in [Utilities for Developers](https://scikit-learn.org/stable/developers/utilities.html#developers-utils).

Here’s a simple example of code using some of the above guidelines:

```
from sklearn.utils import check_array, check_random_state

def choose_random_sample(X, random_state=0):
    """Choose a random point from X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        An array representing the data.
    random_state : int or RandomState instance, default=0
        The seed of the pseudo random number generator that selects a
        random sample. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    x : ndarray of shape (n_features,)
        A random point selected from X.
    """
    X = check_array(X)
    random_state = check_random_state(random_state)
    i = random_state.randint(X.shape[0])
    return X[i]
```

If you use randomness in an estimator instead of a freestanding function, some additional guidelines apply.

First off, the estimator should take a `random_state` argument to its `__init__` with a default value of `None`. It should store that argument’s value, **unmodified**, in an attribute `random_state`. `fit` can call `check_random_state` on that attribute to get an actual random number generator. If, for some reason, randomness is needed after `fit`, the RNG should be stored in an attribute `random_state_`. The following example should make this clear:

```
class GaussianNoise(BaseEstimator, TransformerMixin):
    """This estimator ignores its input and returns random Gaussian noise.

    It also does not adhere to all scikit-learn conventions,
    but showcases how to handle randomness.
    """

    def __init__(self, n_components=100, random_state=None):
        self.random_state = random_state
        self.n_components = n_components

    # the arguments are ignored anyway, so we make them optional
    def fit(self, X=None, y=None):
        self.random_state_ = check_random_state(self.random_state)

    def transform(self, X):
        n_samples = X.shape[0]
        return self.random_state_.randn(n_samples, self.n_components)
```

The reason for this setup is reproducibility: when an estimator is `fit` twice to the same data, it should produce an identical model both times, hence the validation in `fit`, not `__init__`.

#### Numerical assertions in tests

When asserting the quasi-equality of arrays of continuous values, do use `sklearn.utils._testing.assert_allclose`.

The relative tolerance is automatically inferred from the provided arrays dtypes (for float32 and float64 dtypes in particular) but you can override via `rtol`.

When comparing arrays of zero-elements, please do provide a non-zero value for the absolute tolerance via `atol`.

For more information, please refer to the docstring of `sklearn.utils._testing.assert_allclose`.

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/tips.html>

## Developers’ Tips and Tricks

### Productivity and sanity-preserving tips

In this section we gather some useful advice and tools that may increase your quality-of-life when reviewing pull requests, running unit tests, and so forth. Some of these tricks consist of userscripts that require a browser extension such as [TamperMonkey](https://tampermonkey.net/) or [GreaseMonkey](https://www.greasespot.net/); to set up userscripts you must have one of these extensions installed, enabled and running. We provide userscripts as GitHub gists; to install them, click on the “Raw” button on the gist page.

#### Folding and unfolding outdated diffs on pull requests

GitHub hides discussions on PRs when the corresponding lines of code have been changed in the meantime. This [userscript](https://raw.githubusercontent.com/lesteve/userscripts/master/github-expand-all.user.js) provides a shortcut (Control-Alt-P at the time of writing but look at the code to be sure) to unfold all such hidden discussions at once, so you can catch up.

#### Checking out pull requests as remote-tracking branches

In your local fork, add to your `.git/config`, under the `[remote "upstream"]` heading, the line:

```
fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
```

You may then use `git checkout pr/PR_NUMBER` to navigate to the code of the pull-request with the given number. ([Read more in this gist.](https://gist.github.com/piscisaureus/3342247))

#### Display code coverage in pull requests

To overlay the code coverage reports generated by the CodeCov continuous integration, consider [this browser extension](https://github.com/codecov/browser-extension). The coverage of each line will be displayed as a color background behind the line number.

#### Useful pytest aliases and flags

The full test suite takes fairly long to run. For faster iterations, it is possible to select a subset of tests using pytest selectors. In particular, one can run a [single test based on its node ID](https://docs.pytest.org/en/latest/example/markers.html#selecting-tests-based-on-their-node-id):

```
pytest -v sklearn/linear_model/tests/test_logistic.py::test_sparsify
```

or use the [-k pytest parameter](https://docs.pytest.org/en/latest/example/markers.html#using-k-expr-to-select-tests-based-on-their-name) to select tests based on their name. For instance,:

```
pytest sklearn/tests/test_common.py -v -k LogisticRegression
```

will run all [common tests](https://scikit-learn.org/stable/glossary.html#term-common-tests) for the `LogisticRegression` estimator.

When a unit test fails, the following tricks can make debugging easier:

1. The command line argument `pytest -l` instructs pytest to print the local variables when a failure occurs.

1. The argument `pytest --pdb` drops into the Python debugger on failure. To instead drop into the rich IPython debugger `ipdb`, you may set up a shell alias to:

   ```
   pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --capture no
   ```

Other `pytest` options that may become useful include:

- `-x` which exits on the first failed test,

- `--lf` to rerun the tests that failed on the previous run,

- `--ff` to rerun all previous tests, running the ones that failed first,

- `-s` so that pytest does not capture the output of `print()` statements,

- `--tb=short` or `--tb=line` to control the length of the logs,

- `--runxfail` also run tests marked as a known failure (XFAIL) and report errors.

Since our continuous integration tests will error if `FutureWarning` isn’t properly caught, it is also recommended to run `pytest` along with the `-Werror::FutureWarning` flag.

#### Standard replies for reviewing

It may be helpful to store some of these in GitHub’s [saved replies](https://github.com/settings/replies/) for reviewing:

Issue: Usage questions

```
You are asking a usage question. The issue tracker is for bugs and new features. For usage questions, it is recommended to try [Stack Overflow](https://stackoverflow.com/questions/tagged/scikit-learn) or [the Mailing List](https://mail.python.org/mailman/listinfo/scikit-learn).

Unfortunately, we need to close this issue as this issue tracker is a communication tool used for the development of scikit-learn. The additional activity created by usage questions crowds it too much and impedes this development. The conversation can continue here, however there is no guarantee that it will receive attention from core developers.
```

Issue: You’re welcome to update the docs

```
Please feel free to offer a pull request updating the documentation if you feel it could be improved.
```

Issue: Self-contained example for bug

```
Please provide [self-contained example code](https://scikit-learn.org/dev/developers/minimal_reproducer.html), including imports and data (if possible), so that other contributors can just run it and reproduce your issue. Ideally your example code should be minimal.
```

Issue: Software versions

````
To help diagnose your issue, please paste the output of:
```py
import sklearn; sklearn.show_versions()
```
Thanks.
````

Issue: Code blocks

````
Readability can be greatly improved if you [format](https://help.github.com/articles/creating-and-highlighting-code-blocks/) your code snippets and complete error messages appropriately. For example:

    ```python
    print(something)
    ```

generates:

```python
print(something)
```

And:

    ```pytb
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ImportError: No module named 'hello'
    ```

generates:

```pytb
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
ImportError: No module named 'hello'
```

You can edit your issue descriptions and comments at any time to improve readability. This helps maintainers a lot. Thanks!
````

Issue/Comment: Linking to code

```
Friendly advice: for clarity's sake, you can link to code like [this](https://help.github.com/articles/creating-a-permanent-link-to-a-code-snippet/).
```

Issue/Comment: Linking to comments

```
Please use links to comments, which make it a lot easier to see what you are referring to, rather than just linking to the issue. See [this](https://stackoverflow.com/questions/25163598/how-do-i-reference-a-specific-issue-comment-on-github) for more details.
```

PR-NEW: Better description and title

```
Thanks for the pull request! Please make the title of the PR more descriptive. The title will become the commit message when this is merged. You should state what issue (or PR) it fixes/resolves in the description using the syntax described [here](https://scikit-learn.org/dev/developers/contributing.html#contributing-pull-requests).
```

PR-NEW: Fix #

```
Please use "Fix #issueNumber" in your PR description (and you can do it more than once). This way the associated issue gets closed automatically when the PR is merged. For more details, look at [this](https://github.com/blog/1506-closing-issues-via-pull-requests).
```

PR-NEW or Issue: Maintenance cost

```
Every feature we include has a [maintenance cost](https://scikit-learn.org/dev/faq.html#why-are-you-so-selective-on-what-algorithms-you-include-in-scikit-learn). Our maintainers are mostly volunteers. For a new feature to be included, we need evidence that it is often useful and, ideally, [well-established](https://scikit-learn.org/dev/faq.html#what-are-the-inclusion-criteria-for-new-algorithms) in the literature or in practice. Also, we expect PR authors to take part in the maintenance for the code they submit, at least initially. That doesn't stop you implementing it for yourself and publishing it in a separate repository, or even [scikit-learn-contrib](https://scikit-learn-contrib.github.io).
```

PR-WIP: What’s needed before merge?

```
Please clarify (perhaps as a TODO list in the PR description) what work you believe still needs to be done before it can be reviewed for merge. When it is ready, please prefix the PR title with `[MRG]`.
```

PR-WIP: Regression test needed

```
Please add a [non-regression test](https://en.wikipedia.org/wiki/Non-regression_testing) that would fail at main but pass in this PR.
```

PR-MRG: Patience

```
Before merging, we generally require two core developers to agree that your pull request is desirable and ready. [Please be patient](https://scikit-learn.org/dev/faq.html#why-is-my-pull-request-not-getting-any-attention), as we mostly rely on volunteered time from busy core developers. (You are also welcome to help us out with [reviewing other PRs](https://scikit-learn.org/dev/developers/contributing.html#code-review-guidelines).)
```

PR-MRG: Add to what’s new

```
Please add an entry to the future changelog by adding an RST fragment into the module associated with your change located in `doc/whats_new/upcoming_changes`. Refer to the following [README](https://github.com/scikit-learn/scikit-learn/blob/main/doc/whats_new/upcoming_changes/README.md) for full instructions.
```

PR: Don’t change unrelated

```
Please do not change unrelated lines. It makes your contribution harder to review and may introduce merge conflicts to other pull requests.
```

#### Debugging CI issues

CI issues may arise for a variety of reasons, so this is by no means a comprehensive guide, but rather a list of useful tips and tricks.

##### Using a lock-file to get an environment close to the CI

`conda-lock` can be used to create a conda environment with the exact same conda and pip packages as on the CI. For example, the following command will create a conda environment named `scikit-learn-doc` that is similar to the CI:

```
conda-lock install -n scikit-learn-doc build_tools/circle/doc_linux-64_conda.lock
```

Note

It only works if you have the same OS as the CI build (check `platform:` in the lock-file). For example, the previous command will only work if you are on a Linux machine. Also this may not allow you to reproduce some of the issues that are more tied to the particularities of the CI environment, for example CPU architecture reported by OpenBLAS in `sklearn.show_versions()`.

If you don’t have the same OS as the CI build you can still create a conda environment from the right environment yaml file, although it won’t be as close as the CI environment as using the associated lock-file. For example for the doc build:

```
conda env create -n scikit-learn-doc -f build_tools/circle/doc_environment.yml -y
```

This may not give you exactly the same package versions as in the CI for a variety of reasons, for example:

- some packages may have had new releases between the time the lock files were last updated in the `main` branch and the time you run the `conda create` command. You can always try to look at the version in the lock-file and specify the versions by hand for some specific packages that you think would help reproducing the issue.

- different packages may be installed by default depending on the OS. For example, the default BLAS library when installing numpy is OpenBLAS on Linux and MKL on Windows.

Also the problem may be OS specific so the only way to be able to reproduce would be to have the same OS as the CI build.

### Debugging memory errors in Cython with valgrind

While python/numpy’s built-in memory management is relatively robust, it can lead to performance penalties for some routines. For this reason, much of the high-performance code in scikit-learn is written in cython. This performance gain comes with a tradeoff, however: it is very easy for memory bugs to crop up in cython code, especially in situations where that code relies heavily on pointer arithmetic.

Memory errors can manifest themselves a number of ways. The easiest ones to debug are often segmentation faults and related glibc errors. Uninitialized variables can lead to unexpected behavior that is difficult to track down. A very useful tool when debugging these sorts of errors is [valgrind](https://valgrind.org).

Valgrind is a command-line tool that can trace memory errors in a variety of code. Follow these steps:

1. Install [valgrind](https://valgrind.org) on your system.

1. Download the python valgrind suppression file: [valgrind-python.supp](https://github.com/python/cpython/blob/master/Misc/valgrind-python.supp).

1. Follow the directions in the [README.valgrind](https://github.com/python/cpython/blob/master/Misc/README.valgrind) file to customize your python suppressions. If you don’t, you will have spurious output coming related to the python interpreter instead of your own code.

1. Run valgrind as follows:

   ```
   valgrind -v --suppressions=valgrind-python.supp python my_test_script.py
   ```

The result will be a list of all the memory-related errors, which reference lines in the C-code generated by cython from your .pyx file. If you examine the referenced lines in the .c file, you will see comments which indicate the corresponding location in your .pyx source file. Hopefully the output will give you clues as to the source of your memory error.

For more information on valgrind and the array of options it has, see the tutorials and documentation on the [valgrind web site](https://valgrind.org).

### Building and testing for the ARM64 platform on an x86_64 machine

ARM-based machines are a popular target for mobile, edge or other low-energy deployments (including in the cloud, for instance on Scaleway or AWS Graviton).

Here are instructions to setup a local dev environment to reproduce ARM-specific bugs or test failures on an x86_64 host laptop or workstation. This is based on QEMU user mode emulation using docker for convenience (see [multiarch/qemu-user-static](https://github.com/multiarch/qemu-user-static)).

Note

The following instructions are illustrated for ARM64 but they also apply to ppc64le, after changing the Docker image and Miniforge paths appropriately.

Prepare a folder on the host filesystem and download the necessary tools and source code:

```
mkdir arm64
pushd arm64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
git clone https://github.com/scikit-learn/scikit-learn.git
```

Use docker to install QEMU user mode and run an ARM64v8 container with access to your shared folder under the `/io` mount point:

```
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker run -v `pwd`:/io --rm -it arm64v8/ubuntu /bin/bash
```

In the container, install miniforge3 for the ARM64 (a.k.a. aarch64) architecture:

```
bash Miniforge3-Linux-aarch64.sh
# Choose to install miniforge3 under: `/io/miniforge3`
```

Whenever you restart a new container, you will need to reinit the conda env previously installed under `/io/miniforge3`:

```
/io/miniforge3/bin/conda init
source /root/.bashrc
```

as the `/root` home folder is part of the ephemeral docker container. Every file or directory stored under `/io` is persistent on the other hand.

You can then build scikit-learn as usual (you will need to install compiler tools and dependencies using apt or conda as usual). Building scikit-learn takes a lot of time because of the emulation layer, however it needs to be done only once if you put the scikit-learn folder under the `/io` mount point.

Then use pytest to run only the tests of the module you are interested in debugging.

### The Meson Build Backend

Since scikit-learn 1.5.0 we use meson-python as the build tool. Meson is a new tool for scikit-learn and the PyData ecosystem. It is used by several other packages that have written good guides about what it is and how it works.

- [pandas setup doc](https://pandas.pydata.org/docs/development/contributing_environment.html#step-3-build-and-install-pandas): pandas has a similar setup as ours (no spin or dev.py)

- [scipy Meson doc](https://scipy.github.io/devdocs/building/understanding_meson.html) gives more background about how Meson works behind the scenes

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/utilities.html>

## Utilities for Developers

Scikit-learn contains a number of utilities to help with development. These are located in [`sklearn.utils`](https://scikit-learn.org/stable/api/sklearn.utils.html#module-sklearn.utils "sklearn.utils"), and include tools in a number of categories. All the following functions and classes are in the module [`sklearn.utils`](https://scikit-learn.org/stable/api/sklearn.utils.html#module-sklearn.utils "sklearn.utils").

Warning

These utilities are meant to be used internally within the scikit-learn package. They are not guaranteed to be stable between versions of scikit-learn. Backports, in particular, will be removed as the scikit-learn dependencies evolve.

### Validation Tools

These are tools used to check and validate input. When you write a function which accepts arrays, matrices, or sparse matrices as arguments, the following should be used when applicable.

- [`assert_all_finite`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite "sklearn.utils.assert_all_finite"): Throw an error if array contains NaNs or Infs.

- [`as_float_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.as_float_array.html#sklearn.utils.as_float_array "sklearn.utils.as_float_array"): convert input to an array of floats. If a sparse matrix is passed, a sparse matrix will be returned.

- [`check_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array "sklearn.utils.check_array"): check that input is a 2D array, raise error on sparse matrices. Allowed sparse matrix formats can be given optionally, as well as allowing 1D or N-dimensional arrays. Calls [`assert_all_finite`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite "sklearn.utils.assert_all_finite") by default.

- [`check_X_y`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y "sklearn.utils.check_X_y"): check that X and y have consistent length, calls check_array on X, and column_or_1d on y. For multilabel classification or multitarget regression, specify multi_output=True, in which case check_array will be called on y.

- [`indexable`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.indexable.html#sklearn.utils.indexable "sklearn.utils.indexable"): check that all input arrays have consistent length and can be sliced or indexed using safe_index. This is used to validate input for cross-validation.

- [`validation.check_memory`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_memory.html#sklearn.utils.validation.check_memory "sklearn.utils.validation.check_memory") checks that input is `joblib.Memory`-like, which means that it can be converted into a `sklearn.utils.Memory` instance (typically a str denoting the `cachedir`) or has the same interface.

If your code relies on a random number generator, it should never use functions like `numpy.random.random` or `numpy.random.normal`. This approach can lead to repeatability issues in unit tests. Instead, a `numpy.random.RandomState` object should be used, which is built from a `random_state` argument passed to the class or function. The function [`check_random_state`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state "sklearn.utils.check_random_state"), below, can then be used to create a random number generator object.

- [`check_random_state`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state "sklearn.utils.check_random_state"): create a `np.random.RandomState` object from a parameter `random_state`.

  - If `random_state` is `None` or `np.random`, then a randomly-initialized `RandomState` object is returned.

  - If `random_state` is an integer, then it is used to seed a new `RandomState` object.

  - If `random_state` is a `RandomState` object, then it is passed through.

For example:

```
>>> from sklearn.utils import check_random_state
>>> random_state = 0
>>> random_state = check_random_state(random_state)
>>> random_state.rand(4)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
```

When developing your own scikit-learn compatible estimator, the following helpers are available.

- [`validation.check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted "sklearn.utils.validation.check_is_fitted"): check that the estimator has been fitted before calling `transform`, `predict`, or similar methods. This helper allows to raise a standardized error message across estimator.

- [`validation.has_fit_parameter`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.has_fit_parameter.html#sklearn.utils.validation.has_fit_parameter "sklearn.utils.validation.has_fit_parameter"): check that a given parameter is supported in the `fit` method of a given estimator.

### Efficient Linear Algebra & Array Operations

- [`extmath.randomized_range_finder`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_range_finder.html#sklearn.utils.extmath.randomized_range_finder "sklearn.utils.extmath.randomized_range_finder"): construct an orthonormal matrix whose range approximates the range of the input. This is used in [`extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd "sklearn.utils.extmath.randomized_svd"), below.

- [`extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd "sklearn.utils.extmath.randomized_svd"): compute the k-truncated randomized SVD. This algorithm finds the exact truncated singular values decomposition using randomization to speed up the computations. It is particularly fast on large matrices on which you wish to extract only a small number of components.

- `arrayfuncs.cholesky_delete`: (used in [`lars_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path "sklearn.linear_model.lars_path")) Remove an item from a cholesky factorization.

- [`arrayfuncs.min_pos`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.min_pos.html#sklearn.utils.arrayfuncs.min_pos "sklearn.utils.arrayfuncs.min_pos"): (used in `sklearn.linear_model.least_angle`) Find the minimum of the positive values within an array.

- [`extmath.fast_logdet`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.fast_logdet.html#sklearn.utils.extmath.fast_logdet "sklearn.utils.extmath.fast_logdet"): efficiently compute the log of the determinant of a matrix.

- [`extmath.density`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.density.html#sklearn.utils.extmath.density "sklearn.utils.extmath.density"): efficiently compute the density of a sparse vector

- [`extmath.safe_sparse_dot`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html#sklearn.utils.extmath.safe_sparse_dot "sklearn.utils.extmath.safe_sparse_dot"): dot product which will correctly handle `scipy.sparse` inputs. If the inputs are dense, it is equivalent to `numpy.dot`.

- [`extmath.weighted_mode`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html#sklearn.utils.extmath.weighted_mode "sklearn.utils.extmath.weighted_mode"): an extension of `scipy.stats.mode` which allows each item to have a real-valued weight.

- [`resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample "sklearn.utils.resample"): Resample arrays or sparse matrices in a consistent way. used in [`shuffle`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle "sklearn.utils.shuffle"), below.

- [`shuffle`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle "sklearn.utils.shuffle"): Shuffle arrays or sparse matrices in a consistent way. Used in [`k_means`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means "sklearn.cluster.k_means").

### Efficient Random Sampling

- [`random.sample_without_replacement`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html#sklearn.utils.random.sample_without_replacement "sklearn.utils.random.sample_without_replacement"): implements efficient algorithms for sampling `n_samples` integers from a population of size `n_population` without replacement.

### Efficient Routines for Sparse Matrices

The `sklearn.utils.sparsefuncs` cython module hosts compiled extensions to efficiently process `scipy.sparse` data.

- [`sparsefuncs.mean_variance_axis`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.mean_variance_axis.html#sklearn.utils.sparsefuncs.mean_variance_axis "sklearn.utils.sparsefuncs.mean_variance_axis"): compute the means and variances along a specified axis of a CSR matrix. Used for normalizing the tolerance stopping criterion in [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans "sklearn.cluster.KMeans").

- [`sparsefuncs_fast.inplace_csr_row_normalize_l1`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1 "sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1") and [`sparsefuncs_fast.inplace_csr_row_normalize_l2`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2 "sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2"): can be used to normalize individual sparse samples to unit L1 or L2 norm as done in [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer "sklearn.preprocessing.Normalizer").

- [`sparsefuncs.inplace_csr_column_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_csr_column_scale.html#sklearn.utils.sparsefuncs.inplace_csr_column_scale "sklearn.utils.sparsefuncs.inplace_csr_column_scale"): can be used to multiply the columns of a CSR matrix by a constant scale (one scale per column). Used for scaling features to unit standard deviation in [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler "sklearn.preprocessing.StandardScaler").

- [`sort_graph_by_row_values`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.sort_graph_by_row_values.html#sklearn.neighbors.sort_graph_by_row_values "sklearn.neighbors.sort_graph_by_row_values"): can be used to sort a CSR sparse matrix such that each row is stored with increasing values. This is useful to improve efficiency when using precomputed sparse distance matrices in estimators relying on nearest neighbors graph.

### Graph Routines

- [`graph.single_source_shortest_path_length`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph.single_source_shortest_path_length.html#sklearn.utils.graph.single_source_shortest_path_length "sklearn.utils.graph.single_source_shortest_path_length"): (not currently used in scikit-learn) Return the shortest path from a single source to all connected nodes on a graph. Code is adapted from [networkx](https://networkx.github.io/). If this is ever needed again, it would be far faster to use a single iteration of Dijkstra’s algorithm from `graph_shortest_path`.

### Testing Functions

- [`discovery.all_estimators`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_estimators.html#sklearn.utils.discovery.all_estimators "sklearn.utils.discovery.all_estimators") : returns a list of all estimators in scikit-learn to test for consistent behavior and interfaces.

- [`discovery.all_displays`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_displays.html#sklearn.utils.discovery.all_displays "sklearn.utils.discovery.all_displays") : returns a list of all displays (related to plotting API) in scikit-learn to test for consistent behavior and interfaces.

- [`discovery.all_functions`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_functions.html#sklearn.utils.discovery.all_functions "sklearn.utils.discovery.all_functions") : returns a list of all functions in scikit-learn to test for consistent behavior and interfaces.

### Multiclass and multilabel utility function

- [`multiclass.is_multilabel`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.is_multilabel.html#sklearn.utils.multiclass.is_multilabel "sklearn.utils.multiclass.is_multilabel"): Helper function to check if the task is a multi-label classification one.

- [`multiclass.unique_labels`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.unique_labels.html#sklearn.utils.multiclass.unique_labels "sklearn.utils.multiclass.unique_labels"): Helper function to extract an ordered array of unique labels from different formats of target.

### Helper Functions

- [`gen_even_slices`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_even_slices.html#sklearn.utils.gen_even_slices "sklearn.utils.gen_even_slices"): generator to create `n`-packs of slices going up to `n`. Used in [`dict_learning`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning.html#sklearn.decomposition.dict_learning "sklearn.decomposition.dict_learning") and [`k_means`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means "sklearn.cluster.k_means").

- [`gen_batches`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_batches.html#sklearn.utils.gen_batches "sklearn.utils.gen_batches"): generator to create slices containing batch size elements from 0 to `n`

- [`safe_mask`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_mask.html#sklearn.utils.safe_mask "sklearn.utils.safe_mask"): Helper function to convert a mask to the format expected by the numpy array or scipy sparse matrix on which to use it (sparse matrices support integer indices only while numpy arrays support both boolean masks and integer indices).

- [`safe_sqr`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_sqr.html#sklearn.utils.safe_sqr "sklearn.utils.safe_sqr"): Helper function for unified squaring (`**2`) of array-likes, matrices and sparse matrices.

### Hash Functions

- [`murmurhash3_32`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.murmurhash3_32.html#sklearn.utils.murmurhash3_32 "sklearn.utils.murmurhash3_32") provides a python wrapper for the `MurmurHash3_x86_32` C++ non cryptographic hash function. This hash function is suitable for implementing lookup tables, Bloom filters, Count Min Sketch, feature hashing and implicitly defined sparse random projections:

  ```
  >>> from sklearn.utils import murmurhash3_32
  >>> murmurhash3_32("some feature", seed=0) == -384616559
  True

  >>> murmurhash3_32("some feature", seed=0, positive=True) == 3910350737
  True
  ```

  The `sklearn.utils.murmurhash` module can also be “cimported” from other cython modules so as to benefit from the high performance of MurmurHash while skipping the overhead of the Python interpreter.

### Warnings and Exceptions

- [`deprecated`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated "sklearn.utils.deprecated"): Decorator to mark a function or class as deprecated.

- [`ConvergenceWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html#sklearn.exceptions.ConvergenceWarning "sklearn.exceptions.ConvergenceWarning"): Custom warning to catch convergence problems. Used in `sklearn.covariance.graphical_lasso`.

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/performance.html>

## How to optimize for speed

The following gives some practical guidelines to help you write efficient code for the scikit-learn project.

Note

While it is always useful to profile your code so as to **check performance assumptions**, it is also highly recommended to **review the literature** to ensure that the implemented algorithm is the state of the art for the task before investing into costly implementation optimization.

Times and times, hours of efforts invested in optimizing complicated implementation details have been rendered irrelevant by the subsequent discovery of simple **algorithmic tricks**, or by using another algorithm altogether that is better suited to the problem.

The section [A simple algorithmic trick: warm restarts](https://scikit-learn.org/stable/developers/performance.html#warm-restarts) gives an example of such a trick.

### Python, Cython or C/C++?

In general, the scikit-learn project emphasizes the **readability** of the source code to make it easy for the project users to dive into the source code so as to understand how the algorithm behaves on their data but also for ease of maintainability (by the developers).

When implementing a new algorithm is thus recommended to **start implementing it in Python using Numpy and Scipy** by taking care of avoiding looping code using the vectorized idioms of those libraries. In practice this means trying to **replace any nested for loops by calls to equivalent Numpy array methods**. The goal is to avoid the CPU wasting time in the Python interpreter rather than crunching numbers to fit your statistical model. It’s generally a good idea to consider NumPy and SciPy performance tips: [https://scipy.github.io/old-wiki/pages/PerformanceTips](https://scipy.github.io/old-wiki/pages/PerformanceTips)

Sometimes however an algorithm cannot be expressed efficiently in simple vectorized Numpy code. In this case, the recommended strategy is the following:

1. **Profile** the Python implementation to find the main bottleneck and isolate it in a **dedicated module level function**. This function will be reimplemented as a compiled extension module.

1. If there exists a well maintained BSD or MIT **C/C++** implementation of the same algorithm that is not too big, you can write a **Cython wrapper** for it and include a copy of the source code of the library in the scikit-learn source tree: this strategy is used for the classes [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC"), [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC "sklearn.svm.SVC") and [`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") (wrappers for liblinear and libsvm).

1. Otherwise, write an optimized version of your Python function using **Cython** directly. This strategy is used for the [`linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet "sklearn.linear_model.ElasticNet") and [`linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier") classes for instance.

1. **Move the Python version of the function in the tests** and use it to check that the results of the compiled extension are consistent with the gold standard, easy to debug Python version.

1. Once the code is optimized (not simple bottleneck spottable by profiling), check whether it is possible to have **coarse grained parallelism** that is amenable to **multi-processing** by using the `joblib.Parallel` class.

### Profiling Python code

In order to profile Python code we recommend to write a script that loads and prepare you data and then use the IPython integrated profiler for interactively exploring the relevant part for the code.

Suppose we want to profile the Non Negative Matrix Factorization module of scikit-learn. Let us setup a new IPython session and load the digits dataset and as in the [Recognizing hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py) example:

```
In [1]: from sklearn.decomposition import NMF

In [2]: from sklearn.datasets import load_digits

In [3]: X, _ = load_digits(return_X_y=True)
```

Before starting the profiling session and engaging in tentative optimization iterations, it is important to measure the total execution time of the function we want to optimize without any kind of profiler overhead and save it somewhere for later reference:

```
In [4]: %timeit NMF(n_components=16, tol=1e-2).fit(X)
1 loops, best of 3: 1.7 s per loop
```

To have a look at the overall performance profile using the `%prun` magic command:

```
In [5]: %prun -l nmf.py NMF(n_components=16, tol=1e-2).fit(X)
         14496 function calls in 1.682 CPU seconds

   Ordered by: internal time
   List reduced from 90 to 9 due to restriction <'nmf.py'>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       36    0.609    0.017    1.499    0.042 nmf.py:151(_nls_subproblem)
     1263    0.157    0.000    0.157    0.000 nmf.py:18(_pos)
        1    0.053    0.053    1.681    1.681 nmf.py:352(fit_transform)
      673    0.008    0.000    0.057    0.000 nmf.py:28(norm)
        1    0.006    0.006    0.047    0.047 nmf.py:42(_initialize_nmf)
       36    0.001    0.000    0.010    0.000 nmf.py:36(_sparseness)
       30    0.001    0.000    0.001    0.000 nmf.py:23(_neg)
        1    0.000    0.000    0.000    0.000 nmf.py:337(__init__)
        1    0.000    0.000    1.681    1.681 nmf.py:461(fit)
```

The `tottime` column is the most interesting: it gives the total time spent executing the code of a given function ignoring the time spent in executing the sub-functions. The real total time (local code + sub-function calls) is given by the `cumtime` column.

Note the use of the `-l nmf.py` that restricts the output to lines that contain the “nmf.py” string. This is useful to have a quick look at the hotspot of the nmf Python module itself ignoring anything else.

Here is the beginning of the output of the same command without the `-l nmf.py` filter:

```
In [5] %prun NMF(n_components=16, tol=1e-2).fit(X)
         16159 function calls in 1.840 CPU seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     2833    0.653    0.000    0.653    0.000 {numpy.core._dotblas.dot}
       46    0.651    0.014    1.636    0.036 nmf.py:151(_nls_subproblem)
     1397    0.171    0.000    0.171    0.000 nmf.py:18(_pos)
     2780    0.167    0.000    0.167    0.000 {method 'sum' of 'numpy.ndarray' objects}
        1    0.064    0.064    1.840    1.840 nmf.py:352(fit_transform)
     1542    0.043    0.000    0.043    0.000 {method 'flatten' of 'numpy.ndarray' objects}
      337    0.019    0.000    0.019    0.000 {method 'all' of 'numpy.ndarray' objects}
     2734    0.011    0.000    0.181    0.000 fromnumeric.py:1185(sum)
        2    0.010    0.005    0.010    0.005 {numpy.linalg.lapack_lite.dgesdd}
      748    0.009    0.000    0.065    0.000 nmf.py:28(norm)
...
```

The above results show that the execution is largely dominated by dot product operations (delegated to blas). Hence there is probably no huge gain to expect by rewriting this code in Cython or C/C++: in this case out of the 1.7s total execution time, almost 0.7s are spent in compiled code we can consider optimal. By rewriting the rest of the Python code and assuming we could achieve a 1000% boost on this portion (which is highly unlikely given the shallowness of the Python loops), we would not gain more than a 2.4x speed-up globally.

Hence major improvements can only be achieved by **algorithmic improvements** in this particular example (e.g. trying to find operations that are both costly and useless to avoid computing them rather than trying to optimize their implementation).

It is however still interesting to check what’s happening inside the `_nls_subproblem` function which is the hotspot if we only consider Python code: it takes around 100% of the accumulated time of the module. In order to better understand the profile of this specific function, let us install `line_profiler` and wire it to IPython:

```
pip install line_profiler
```

**Under IPython 0.13+**, first create a configuration profile:

```
ipython profile create
```

Then register the line_profiler extension in `~/.ipython/profile_default/ipython_config.py`:

```
c.TerminalIPythonApp.extensions.append('line_profiler')
c.InteractiveShellApp.extensions.append('line_profiler')
```

This will register the `%lprun` magic command in the IPython terminal application and the other frontends such as qtconsole and notebook.

Now restart IPython and let us use this new toy:

```
In [1]: from sklearn.datasets import load_digits

In [2]: from sklearn.decomposition import NMF
  ... : from sklearn.decomposition._nmf import _nls_subproblem

In [3]: X, _ = load_digits(return_X_y=True)

In [4]: %lprun -f _nls_subproblem NMF(n_components=16, tol=1e-2).fit(X)
Timer unit: 1e-06 s

File: sklearn/decomposition/nmf.py
Function: _nls_subproblem at line 137
Total time: 1.73153 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   137                                           def _nls_subproblem(V, W, H_init, tol, max_iter):
   138                                               """Non-negative least square solver
   ...
   170                                               """
   171        48         5863    122.1      0.3      if (H_init < 0).any():
   172                                                   raise ValueError("Negative values in H_init passed to NLS solver.")
   173
   174        48          139      2.9      0.0      H = H_init
   175        48       112141   2336.3      5.8      WtV = np.dot(W.T, V)
   176        48        16144    336.3      0.8      WtW = np.dot(W.T, W)
   177
   178                                               # values justified in the paper
   179        48          144      3.0      0.0      alpha = 1
   180        48          113      2.4      0.0      beta = 0.1
   181       638         1880      2.9      0.1      for n_iter in range(1, max_iter + 1):
   182       638       195133    305.9     10.2          grad = np.dot(WtW, H) - WtV
   183       638       495761    777.1     25.9          proj_gradient = norm(grad[np.logical_or(grad < 0, H > 0)])
   184       638         2449      3.8      0.1          if proj_gradient < tol:
   185        48          130      2.7      0.0              break
   186
   187      1474         4474      3.0      0.2          for inner_iter in range(1, 20):
   188      1474        83833     56.9      4.4              Hn = H - alpha * grad
   189                                                       # Hn = np.where(Hn > 0, Hn, 0)
   190      1474       194239    131.8     10.1              Hn = _pos(Hn)
   191      1474        48858     33.1      2.5              d = Hn - H
   192      1474       150407    102.0      7.8              gradd = np.sum(grad * d)
   193      1474       515390    349.7     26.9              dQd = np.sum(np.dot(WtW, d) * d)
   ...
```

By looking at the top values of the `% Time` column it is really easy to pin-point the most expensive expressions that would deserve additional care.

### Memory usage profiling

You can analyze in detail the memory usage of any Python code with the help of [memory_profiler](https://pypi.org/project/memory_profiler/). First, install the latest version:

```
pip install -U memory_profiler
```

Then, setup the magics in a manner similar to `line_profiler`.

**Under IPython 0.11+**, first create a configuration profile:

```
ipython profile create
```

Then register the extension in `~/.ipython/profile_default/ipython_config.py` alongside the line profiler:

```
c.TerminalIPythonApp.extensions.append('memory_profiler')
c.InteractiveShellApp.extensions.append('memory_profiler')
```

This will register the `%memit` and `%mprun` magic commands in the IPython terminal application and the other frontends such as qtconsole and notebook.

`%mprun` is useful to examine, line-by-line, the memory usage of key functions in your program. It is very similar to `%lprun`, discussed in the previous section. For example, from the `memory_profiler` `examples` directory:

```
In [1] from example import my_func

In [2] %mprun -f my_func my_func()
Filename: example.py

Line #    Mem usage  Increment   Line Contents
==============================================
     3                           @profile
     4      5.97 MB    0.00 MB   def my_func():
     5     13.61 MB    7.64 MB       a = [1] * (10 ** 6)
     6    166.20 MB  152.59 MB       b = [2] * (2 * 10 ** 7)
     7     13.61 MB -152.59 MB       del b
     8     13.61 MB    0.00 MB       return a
```

Another useful magic that `memory_profiler` defines is `%memit`, which is analogous to `%timeit`. It can be used as follows:

```
In [1]: import numpy as np

In [2]: %memit np.zeros(1e7)
maximum of 3: 76.402344 MB per loop
```

For more details, see the docstrings of the magics, using `%memit?` and `%mprun?`.

### Using Cython

If profiling of the Python code reveals that the Python interpreter overhead is larger by one order of magnitude or more than the cost of the actual numerical computation (e.g. `for` loops over vector components, nested evaluation of conditional expression, scalar arithmetic…), it is probably adequate to extract the hotspot portion of the code as a standalone function in a `.pyx` file, add static type declarations and then use Cython to generate a C program suitable to be compiled as a Python extension module.

The [Cython’s documentation](https://docs.cython.org/) contains a tutorial and reference guide for developing such a module. For more information about developing in Cython for scikit-learn, see [Cython Best Practices, Conventions and Knowledge](https://scikit-learn.org/stable/developers/cython.html#cython).

### Profiling compiled extensions

When working with compiled extensions (written in C/C++ with a wrapper or directly as Cython extension), the default Python profiler is useless: we need a dedicated tool to introspect what’s happening inside the compiled extension itself.

#### Using yep and gperftools

Easy profiling without special compilation options use yep:

- [https://pypi.org/project/yep/](https://pypi.org/project/yep/)

- [https://fa.bianp.net/blog/2011/a-profiler-for-python-extensions](https://fa.bianp.net/blog/2011/a-profiler-for-python-extensions)

#### Using a debugger, gdb

- It is helpful to use `gdb` to debug. In order to do so, one must use a Python interpreter built with debug support (debug symbols and proper optimization). To create a new conda environment (which you might need to deactivate and reactivate after building/installing) with a source-built CPython interpreter:

  ```
  git clone https://github.com/python/cpython.git
  conda create -n debug-scikit-dev
  conda activate debug-scikit-dev
  cd cpython
  mkdir debug
  cd debug
  ../configure --prefix=$CONDA_PREFIX --with-pydebug
  make EXTRA_CFLAGS='-DPy_DEBUG' -j<num_cores>
  make install
  ```

#### Using gprof

In order to profile compiled Python extensions one could use `gprof` after having recompiled the project with `gcc -pg` and using the `python-dbg` variant of the interpreter on debian / ubuntu: however this approach requires to also have `numpy` and `scipy` recompiled with `-pg` which is rather complicated to get working.

Fortunately there exist two alternative profilers that don’t require you to recompile everything.

#### Using valgrind / callgrind / kcachegrind

##### kcachegrind

`yep` can be used to create a profiling report. `kcachegrind` provides a graphical environment to visualize this report:

```
# Run yep to profile some python script
python -m yep -c my_file.py

# open my_file.py.callgrin with kcachegrind
kcachegrind my_file.py.prof
```

Note

`yep` can be executed with the argument `--lines` or `-l` to compile a profiling report ‘line by line’.

### Multi-core parallelism using `joblib.Parallel`

See [joblib documentation](https://joblib.readthedocs.io)

### A simple algorithmic trick: warm restarts

See the glossary entry for [warm_start](https://scikit-learn.org/stable/glossary.html#term-warm_start)

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/cython.html>

## Cython Best Practices, Conventions and Knowledge

This document contains tips to develop Cython code in scikit-learn.

### Tips for developing with Cython in scikit-learn

#### Tips to ease development

- Time spent reading [Cython’s documentation](https://cython.readthedocs.io/en/latest/) is not time lost.

- If you intend to use OpenMP: On MacOS, system’s distribution of `clang` does not implement OpenMP. You can install the `compilers` package available on `conda-forge` which comes with an implementation of OpenMP.

- Activating [checks](https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87) might help. E.g. for activating boundscheck use:

  ```
  export SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES=1
  ```

- [Start from scratch in a notebook](https://cython.readthedocs.io/en/latest/src/quickstart/build.html#using-the-jupyter-notebook) to understand how to use Cython and to get feedback on your work quickly. If you plan to use OpenMP for your implementations in your Jupyter Notebook, do add extra compiler and linkers arguments in the Cython magic.

  ```
  # For GCC and for clang
  %%cython --compile-args=-fopenmp --link-args=-fopenmp
  # For Microsoft's compilers
  %%cython --compile-args=/openmp --link-args=/openmp
  ```

- To debug C code (e.g. a segfault), do use `gdb` with:

  ```
  gdb --ex r --args python ./entrypoint_to_bug_reproducer.py
  ```

- To have access to some value in place to debug in `cdef (nogil)` context, use:

  ```
  with gil:
      print(state_to_print)
  ```

- Note that Cython cannot parse f-strings with `{var=}` expressions, e.g.

  ```
  print(f"{test_val=}")
  ```

- scikit-learn codebase has a lot of non-unified (fused) types (re)definitions. There currently is [ongoing work to simplify and unify that across the codebase](https://github.com/scikit-learn/scikit-learn/issues/25572). For now, make sure you understand which concrete types are used ultimately.

- You might find this alias to compile individual Cython extension handy:

  ```
  # You might want to add this alias to your shell script config.
  alias cythonX="cython -X language_level=3 -X boundscheck=False -X wraparound=False -X initializedcheck=False -X nonecheck=False -X cdivision=True"

  # This generates `source.c` as if you had recompiled scikit-learn entirely.
  cythonX --annotate source.pyx
  ```

- Using the `--annotate` option with this flag allows generating an HTML report of code annotation. This report indicates interactions with the CPython interpreter on a line-by-line basis. Interactions with the CPython interpreter must be avoided as much as possible in the computationally intensive sections of the algorithms. For more information, please refer to [this section of Cython’s tutorial](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#primes)

  ```
  # This generates an HTML report (`source.html`) for `source.c`.
  cythonX --annotate source.pyx
  ```

#### Tips for performance

- Understand the GIL in context for CPython (which problems it solves, what are its limitations) and get a good understanding of when Cython will be mapped to C code free of interactions with CPython, when it will not, and when it cannot (e.g. presence of interactions with Python objects, which include functions). In this regard, [PEP073](https://peps.python.org/pep-0703/) provides a good overview and context and pathways for removal.

- Make sure you have deactivated [checks](https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87).

- Always prefer memoryviews instead of `cnp.ndarray` when possible: memoryviews are lightweight.

- Avoid memoryview slicing: memoryview slicing might be costly or misleading in some cases and we better not use it, even if handling fewer dimensions in some context would be preferable.

- Decorate final classes or methods with `@final` (this allows removing virtual tables when needed)

- Inline methods and functions when it makes sense

- In doubt, read the generated C or C++ code if you can: “The fewer C instructions and indirections for a line of Cython code, the better” is a good rule of thumb.

- `nogil` declarations are just hints: when declaring the `cdef` functions as nogil, it means that they can be called without holding the GIL, but it does not release the GIL when entering them. You have to do that yourself either by passing `nogil=True` to `cython.parallel.prange` explicitly, or by using an explicit context manager:

  ```
  cdef inline void my_func(self) nogil:

      # Some logic interacting with CPython, e.g. allocating arrays via NumPy.

      with nogil:
          # The code here is run as if it were written in C.

      return 0
  ```

  This item is based on [this comment from Stéfan’s Benhel](https://github.com/cython/cython/issues/2798#issuecomment-459971828)

- Direct calls to BLAS routines are possible via interfaces defined in `sklearn.utils._cython_blas`.

#### Using OpenMP

Since scikit-learn can be built without OpenMP, it’s necessary to protect each direct call to OpenMP.

The `_openmp_helpers` module, available in [sklearn/utils/\_openmp_helpers.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_openmp_helpers.pyx) provides protected versions of the OpenMP routines. To use OpenMP routines, they must be `cimported` from this module and not from the OpenMP library directly:

```
from sklearn.utils._openmp_helpers cimport omp_get_max_threads
max_threads = omp_get_max_threads()
```

The parallel loop, `prange`, is already protected by cython and can be used directly from `cython.parallel`.

##### Types

Cython code requires to use explicit types. This is one of the reasons you get a performance boost. In order to avoid code duplication, we have a central place for the most used types in [sklearn/utils/\_typedefs.pxd](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_typedefs.pxd). Ideally you start by having a look there and `cimport` types you need, for example

```
from sklearn.utils._typedefs cimport float32, float64
```

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/misc_info.html>

## Miscellaneous information / Troubleshooting

Here, you find some more advanced notes and troubleshooting tips related to [Set up your development environment](https://scikit-learn.org/stable/developers/development_setup.html#setup-development-environment).

### Notes on OpenMP

Even though the default C compiler on macOS (Apple clang) is confusingly aliased as `/usr/bin/gcc`, it does not directly support OpenMP.

Note

If OpenMP is not supported by the compiler, the build will be done with OpenMP functionalities disabled. This is not recommended since it will force some estimators to run in sequential mode instead of leveraging thread-based parallelism. Setting the `SKLEARN_FAIL_NO_OPENMP` environment variable (before cythonization) will force the build to fail if OpenMP is not supported.

To check if `scikit-learn` has been built correctly with OpenMP, run

```
python -c "import sklearn; sklearn.show_versions()"
```

and check if it contains `Built with OpenMP: True`.

When using conda on Mac, you can also check that the custom compilers are properly installed from conda-forge using the following command:

```
conda list
```

which should include `compilers` and `llvm-openmp`.

The compilers meta-package will automatically set custom environment variables:

```
echo $CC
echo $CXX
echo $CFLAGS
echo $CXXFLAGS
echo $LDFLAGS
```

They point to files and folders from your `sklearn-dev` conda environment (in particular in the `bin/`, `include/` and `lib/` subfolders). For instance `-L/path/to/conda/envs/sklearn-dev/lib` should appear in `LDFLAGS`.

### Notes on Conda

Sometimes it can be necessary to open a new prompt before activating a newly created conda environment.

If you get any conflicting dependency error messages on Mac or Linux, try commenting out any custom conda configuration in the `$HOME/.condarc` file. In particular the `channel_priority: strict` directive is known to cause problems for this setup.

### Note on dependencies for other Linux distributions

When precompiled wheels of the runtime dependencies are not available for your architecture (e.g. **ARM**), you can install the system versions:

```
sudo apt-get install cython3 python3-numpy python3-scipy
```

### Notes on Meson

When [building scikit-learn from source](https://scikit-learn.org/stable/developers/development_setup.html#install-from-source), existing scikit-learn installations and meson builds can lead to conflicts. You can use the `Makefile` provided in the [scikit-learn repository](https://github.com/scikit-learn/scikit-learn/) to remove conflicting builds by calling:

```
make clean
```

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/bug_triaging.html>

## Bug triaging and issue curation

The [issue tracker](https://github.com/scikit-learn/scikit-learn/issues) is important to the communication in the project: it helps developers identify major projects to work on, as well as to discuss priorities. For this reason, it is important to curate it, adding labels to issues and closing issues that are not necessary.

### Working on issues to improve them

Improving issues increases their chances of being successfully resolved. Guidelines on submitting good issues can be found [here](https://scikit-learn.org/stable/developers/contributing.html#filing-bugs). A third party can give useful feedback or even add comments on the issue. The following actions are typically useful:

- documenting issues that are missing elements to reproduce the problem such as code samples

- suggesting better use of code formatting

- suggesting to reformulate the title and description to make them more explicit about the problem to be solved

- linking to related issues or discussions while briefly describing how they are related, for instance “See also #xyz for a similar attempt at this” or “See also #xyz where the same thing happened in SomeEstimator” provides context and helps the discussion.

Fruitful discussions

Online discussions may be harder than it seems at first glance, in particular given that a person new to open-source may have a very different understanding of the process than a seasoned maintainer.

Overall, it is useful to stay positive and assume good will. [The following article](https://gael-varoquaux.info/programming/technical-discussions-are-hard-a-few-tips.html) explores how to lead online discussions in the context of open source.

### Working on PRs to help review

Reviewing code is also encouraged. Contributors and users are welcome to participate in the review process following our [review guidelines](https://scikit-learn.org/stable/developers/contributing.html#code-review).

### Triaging operations for members of the core and contributor experience teams

In addition to the above, members of the core team and the contributor experience team can do the following important tasks:

- Update [labels for issues and PRs](https://scikit-learn.org/stable/developers/contributing.html#issue-tracker-tags): see the list of the [available github labels](https://github.com/scikit-learn/scikit-learn/labels).

- [Determine if a PR must be relabeled as stalled](https://scikit-learn.org/stable/developers/contributing.html#stalled-pull-request) or needs help (this is typically very important in the context of sprints, where the risk is to create many unfinished PRs)

- If a stalled PR is taken over by a newer PR, then label the stalled PR as “Superseded”, leave a comment on the stalled PR linking to the new PR, and likely close the stalled PR.

- Triage issues:

  - **close usage questions** and politely point the reporter to use Stack Overflow instead.

  - **close duplicate issues**, after checking that they are indeed duplicate. Ideally, the original submitter moves the discussion to the older, duplicate issue

  - **close issues that cannot be replicated**, after leaving time (at least a week) to add extra information

[Saved replies](https://scikit-learn.org/stable/developers/tips.html#saved-replies) are useful to gain time and yet be welcoming and polite when triaging.

See the github description for [roles in the organization](https://docs.github.com/en/github/setting-up-and-managing-organizations-and-teams/repository-permission-levels-for-an-organization).

Closing issues: a tough call

When uncertain on whether an issue should be closed or not, it is best to strive for consensus with the original poster, and possibly to seek relevant expertise. However, when the issue is a usage question, or when it has been considered as unclear for many years it should be closed.

### A typical workflow for triaging issues

The following workflow [[1]](https://scikit-learn.org/stable/developers/bug_triaging.html#id2) is a good way to approach issue triaging:

1. Thank the reporter for opening an issue

   The issue tracker is many people’s first interaction with the scikit-learn project itself, beyond just using the library. As such, we want it to be a welcoming, pleasant experience.

1. Is this a usage question? If so close it with a polite message ([here is an example](https://scikit-learn.org/stable/developers/tips.html#saved-replies)).

1. Is the necessary information provided?

   If crucial information (like the version of scikit-learn used), is missing feel free to ask for that and label the issue with “Needs info”.

1. Is this a duplicate issue?

   We have many open issues. If a new issue seems to be a duplicate, point to the original issue. If it is a clear duplicate, or consensus is that it is redundant, close it. Make sure to still thank the reporter, and encourage them to chime in on the original issue, and perhaps try to fix it.

   If the new issue provides relevant information, such as a better or slightly different example, add it to the original issue as a comment or an edit to the original post.

1. Make sure that the title accurately reflects the issue. If you have the necessary permissions edit it yourself if it’s not clear.

1. Is the issue minimal and reproducible?

   For bug reports, we ask that the reporter provide a minimal reproducible example. See [this useful post](https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports) by Matthew Rocklin for a good explanation. If the example is not reproducible, or if it’s clearly not minimal, feel free to ask the reporter if they can provide an example or simplify the provided one. Do acknowledge that writing minimal reproducible examples is hard work. If the reporter is struggling, you can try to write one yourself.

   If a reproducible example is provided, but you see a simplification, add your simpler reproducible example.

1. Add the relevant labels, such as “Documentation” when the issue is about documentation, “Bug” if it is clearly a bug, “Enhancement” if it is an enhancement request, …

   If the issue is clearly defined and the fix seems relatively straightforward, label the issue as “Good first issue”.

   An additional useful step can be to tag the corresponding module e.g. `sklearn.linear_models` when relevant.

1. Remove the “Needs Triage” label from the issue if the label exists.

\[[1](https://scikit-learn.org/stable/developers/bug_triaging.html#id1)\]

Adapted from the pandas project [maintainers guide](https://pandas.pydata.org/docs/development/maintaining.html)

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/maintainer.html>

## Maintainer Information

### Releasing

This section is about preparing a major/minor release, a release candidate (RC), or a bug-fix release. We follow [PEP440](https://www.python.org/dev/peps/pep-0440/) for the version scheme and to indicate different types of releases. Our convention is to follow the “major.minor.micro” scheme, although in practice there is no fundamental difference between major and minor releases and micro releases are bug-fix releases.

We adopted the following release schedule:

- Major/Minor releases every 6 months, usually in May and November. These releases are numbered `X.Y.0` and are preceded by one or more release candidates `X.Y.0rcN`.

- Bug-fix releases are done as needed between major/minor releases and only apply to the last stable version. These releases are numbered `X.Y.Z`.

Preparation

- Confirm that all blockers tagged for the milestone have been resolved, and that other issues tagged for the milestone can be postponed.

- Make sure the deprecations, FIXMEs, and TODOs tagged for the release have been taken care of.

- Make sure that the minimum supported versions of our dependencies have been bumped, see [Guideline for bumping minimum versions of our dependencies](https://scikit-learn.org/stable/developers/maintainer.html#bumping-dependencies-guideline) for details.

- For major/minor final releases, make sure that a *Release Highlights* page has been done as a runnable example and check that its HTML rendering looks correct. It should be linked from the what’s new file for the new version of scikit-learn.

Permissions

- The release manager must be a **maintainer** of the [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) repository to be able to publish on `pypi.org` and `test.pypi.org` (via a manual trigger of a dedicated Github Actions workflow).

- The release manager must be a **maintainer** of the [conda-forge/scikit-learn-feedstock](https://github.com/conda-forge/scikit-learn-feedstock) repository to be able to publish on `conda-forge`. This can be changed by editing the `recipe/meta.yaml` file in the first release pull request.

#### Reference Steps

Major/Minor RC

Suppose that we are preparing the release `1.9.0rc1`.

The first RC ideally counts as a **feature freeze**. Each coming release candidate and the final release afterwards should include only minor documentation changes and bug fixes. Any major enhancement or new feature should be excluded.

- Create the release branch `1.9.X` directly in the main repository, where `X` is really the letter X, **not a placeholder**. The development for the final and subsequent bug-fix releases of `1.9` should also happen under this branch with different tags.

  ```
  git fetch upstream main
  git checkout upstream/main
  git checkout -b 1.9.X
  git push --set-upstream upstream 1.9.X
  ```

- Create a PR targeting the `1.9.X` branch. Copy the following release checklist to the description of this PR to track the progress.

  ```
  * [ ] Update the sklearn dev0 version in main branch
  * [ ] Set the version number in the release branch
  * [ ] Set an upper bound on build dependencies in the release branch
  * [ ] Generate the changelog in the release branch
  * [ ] Check that the wheels for the release can be built successfully
  * [ ] Merge the PR with `[cd build]` commit message to upload wheels to the staging repo
  * [ ] Upload the wheels and source tarball to https://test.pypi.org
  * [ ] Create tag on the main repo
  * [ ] Confirm bot detected at https://github.com/conda-forge/scikit-learn-feedstock
        and wait for merge
  * [ ] Upload the wheels and source tarball to PyPI
  * [ ] Announce on mailing list and on social media platforms (LinkedIn, Bluesky, etc.)
  ```

- Create a PR from `main` and targeting `main` to prepare for the next version. In this PR you need to:

  - Increment the dev0 `__version__` variable in `sklearn/__init__.py`. This means that while we are in the release candidate period, the latest stable is two versions behind the `main` branch, instead of one.

  - Include a new what’s new file under the `doc/whats_new/` directory. Don’t forget to add an entry for this new file in `doc/whats_new.rst`.

  - Change the what’s new file to the newly created one in the `filename` field of the `tool.towncrier` section in `pyproject.toml`.

- In the release branch, change the version number `__version__` in `sklearn/__init__.py` to `1.9.0rc1`.

- Still in the release branch, set or update the upper bound on the build dependencies in the `[build-system]` section of `pyproject.toml`. The goal is to prevent future backward incompatible releases of the dependencies to break the build in the maintenance branch.

  The upper bounds should match the latest already-released minor versions of the dependencies and should allow future micro (bug-fix) versions. For instance, if numpy 2.2.5 is the most recent version, its upper bound should be set to \<2.3.0.

- In the release branch, generate the changelog for the incoming version, i.e., `doc/whats_new/1.9.rst`. During the RC period we want to keep the fragments when we generate the changelog because we’ll generate it again for the final release, including the changes that may happen in between:

  ```
  towncrier build --keep --version 1.9.0
  ```

- Trigger the wheel builder with the `[cd build]` commit marker. See also the [workflow runs of the wheel builder](https://github.com/scikit-learn/scikit-learn/actions/workflows/wheels.yml).

  ```
  git commit --allow-empty -m "[cd build] Trigger wheel builder workflow"
  ```

  Note

  The acronym CD in `[cd build]` stands for [Continuous Delivery](https://en.wikipedia.org/wiki/Continuous_delivery) and refers to the automation used to generate the release artifacts (binary and source packages). This can be seen as an extension to CI which stands for [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration). The CD workflow on GitHub Actions is also used to automatically create nightly builds and publish packages for the development branch of scikit-learn. See also [Installing nightly builds](https://scikit-learn.org/stable/install.html#install-nightly-builds).

- Once all the CD jobs have completed successfully in the PR, merge it with the `[cd build]` marker in the commit message. This time the results will be uploaded to the staging area. You should then be able to upload the generated artifacts (`.tar.gz` and `.whl` files) to [https://test.pypi.org/](https://test.pypi.org/) using the “Run workflow” form for the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml).

  Warning

  This PR should be merged with the rebase mode instead of the usual squash mode because we want to keep the history in the `1.9.X` branch close to the history of the main branch which will help for future bug fix releases.

  In addition if on merging, the last commit, containing the `[cd build]` marker, is empty, the CD jobs won’t be triggered. In this case, you can directly push a commit with the marker in the `1.9.X` branch to trigger them.

- If the steps above went fine, proceed **with caution** to create a new tag for the release. This should be done only when you are almost certain that the release is ready, since adding a new tag to the main repository can trigger certain automated processes.

  ```
  git tag -a 1.9.0rc1  # in the 1.9.X branch
  git push https://github.com/scikit-learn/scikit-learn.git 1.9.0rc1
  ```

  Warning

  Don’t use the github interface for publishing the release as a way to create the tag because it will automatically send notifications to all users that follow the repo even though the website isn’t updated and wheels aren’t uploaded yet.

- Confirm that the bot has detected the tag on the conda-forge feedstock repository [conda-forge/scikit-learn-feedstock](https://github.com/conda-forge/scikit-learn-feedstock). If not, submit a PR for the release, targeting the `rc` branch. Make sure to update the PR such that it will be synchronized with the `main` branch. In particular, backport migrations that may have been added since the last release.

- Trigger the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml) again, but this time to upload the artifacts to the real [https://pypi.org/](https://pypi.org/). To do so, replace `testpypi` with `pypi` in the “Run workflow” form.

  **Alternatively**, it is possible to collect locally the generated binary wheel packages and source tarball and upload them all to PyPI.

  Uploading artifacts from local

  Check out at the release tag and run the following commands.

  ```
  rm -r dist
  python -m pip install -U wheelhouse_uploader twine
  python -m wheelhouse_uploader fetch \
    --version 1.9.0rc1 --local-folder dist scikit-learn \
    https://pypi.anaconda.org/scikit-learn-wheels-staging/simple/scikit-learn/
  ```

  These commands will download all the binary packages accumulated in the [staging area on the anaconda.org hosting service](https://anaconda.org/scikit-learn-wheels-staging/scikit-learn/files) and put them in your local `./dist` folder. Check the contents of the `./dist` folder: it should contain all the wheels along with the source tarball `.tar.gz`. Make sure you do not have developer versions or older versions of the scikit-learn package in that folder. Before uploading to PyPI, you can test uploading to `test.pypi.org` first.

  ```
  twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
  ```

  Then upload everything at once to `pypi.org`.

  ```
  twine upload dist/*
  ```

Major/Minor Final

Suppose that we are preparing the release `1.9.0`.

- Create a new branch from the `main` branch, then start an interactive rebase from `1.9.X` to select the commits that need to be backported:

  ```
  git rebase -i upstream/1.9.X
  ```

  This will open an interactive rebase with the `git-rebase-todo` containing all the latest commits on `main`. At this stage, you have to perform this interactive rebase with at least someone else (to not forget something and to avoid doubts).

  - Do not remove lines but drop commit by replacing `pick` with `drop`.

  - Commits to pick for a bug-fix release are *generally* prefixed with `FIX`, `CI`, and `DOC`. They should at least include all the commits of the merged PRs that were milestoned for this release.

  - Commits to `drop` for a bug-fix release are *generally* prefixed with `FEAT`, `MAINT`, `ENH`, and `API`. Reasons for not including them are to prevent change of behavior (which should only happen in major/minor releases).

  - After having dropped or picked commits, **do not exit** but paste the content of the `git-rebase-todo` message in the PR. This file is located at `.git/rebase-merge/git-rebase-todo`.

  - Save and exit to start the interactive rebase. Resolve merge conflicts when necessary.

- Create a PR targeting the `1.9.X` branch. Copy the following release checklist to the description of this PR to track the progress.

  ```
  * [ ] Set the version number in the release branch
  * [ ] Set an upper bound on build dependencies in the release branch
  * [ ] Generate the changelog in the release branch
  * [ ] Check that the wheels for the release can be built successfully
  * [ ] Merge the PR with `[cd build]` commit message to upload wheels to the staging repo
  * [ ] Upload the wheels and source tarball to https://test.pypi.org
  * [ ] Create tag on the main repo
  * [ ] Confirm bot detected at https://github.com/conda-forge/scikit-learn-feedstock
        and wait for merge
  * [ ] Upload the wheels and source tarball to PyPI
  * [ ] Update news and what's new date in main branch
  * [ ] Backport news and what's new date in release branch
  * [ ] Cleanup the doc repo to free up space
  * [ ] Update symlink for stable in https://github.com/scikit-learn/scikit-learn.github.io
  * [ ] Publish to https://github.com/scikit-learn/scikit-learn/releases
  * [ ] Announce on mailing list and on social media platforms (LinkedIn, Bluesky, etc.)
  * [ ] Update SECURITY.md in main branch
  ```

- In the release branch, change the version number `__version__` in `sklearn/__init__.py` to `1.9.0`.

- Still in the release branch, set or update the upper bound on the build dependencies in the `[build-system]` section of `pyproject.toml`. The goal is to prevent future backward incompatible releases of the dependencies to break the build in the maintenance branch.

  The upper bounds should match the latest already-released minor versions of the dependencies and should allow future micro (bug-fix) versions. For instance, if numpy 2.2.5 is the most recent version, its upper bound should be set to \<2.3.0.

- In the release branch, generate the changelog for the incoming version, i.e., `doc/whats_new/1.9.rst`. For a non RC release, push a commit where you:

  - Generate the changelog, not keeping the fragments.

    ```
    towncrier build --version 1.9.0
    ```

  - Link the release highlights example.

  - Add the list of contributor names. Suppose that the tag of the last release in the previous major/minor version is `1.8.0`, then you can use the following command to retrieve the list of contributor names:

    ```
    git shortlog -s 1.8.0.. |
      cut -f2- |
      sort --ignore-case |
      tr "\n" ";" |
      sed "s/;/, /g;s/, $//" |
      fold -s
    ```

  Then create a PR targeting the `main` branch and cherry-pick this commit there.

- Trigger the wheel builder with the `[cd build]` commit marker. See also the [workflow runs of the wheel builder](https://github.com/scikit-learn/scikit-learn/actions/workflows/wheels.yml).

  ```
  git commit --allow-empty -m "[cd build] Trigger wheel builder workflow"
  ```

  Note

  The acronym CD in `[cd build]` stands for [Continuous Delivery](https://en.wikipedia.org/wiki/Continuous_delivery) and refers to the automation used to generate the release artifacts (binary and source packages). This can be seen as an extension to CI which stands for [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration). The CD workflow on GitHub Actions is also used to automatically create nightly builds and publish packages for the development branch of scikit-learn. See also [Installing nightly builds](https://scikit-learn.org/stable/install.html#install-nightly-builds).

- Once all the CD jobs have completed successfully in the PR, merge it with the `[cd build]` marker in the commit message. This time the results will be uploaded to the staging area. You should then be able to upload the generated artifacts (`.tar.gz` and `.whl` files) to [https://test.pypi.org/](https://test.pypi.org/) using the “Run workflow” form for the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml).

  Warning

  This PR should be merged with the rebase mode instead of the usual squash mode because we want to keep the history in the `1.9.X` branch close to the history of the main branch which will help for future bug fix releases.

  In addition if on merging, the last commit, containing the `[cd build]` marker, is empty, the CD jobs won’t be triggered. In this case, you can directly push a commit with the marker in the `1.9.X` branch to trigger them.

- If the steps above went fine, proceed **with caution** to create a new tag for the release. This should be done only when you are almost certain that the release is ready, since adding a new tag to the main repository can trigger certain automated processes.

  ```
  git tag -a 1.9.0  # in the 1.9.X branch
  git push https://github.com/scikit-learn/scikit-learn.git 1.9.0
  ```

  Warning

  Don’t use the github interface for publishing the release as a way to create the tag because it will automatically send notifications to all users that follow the repo even though the website isn’t updated and wheels aren’t uploaded yet.

- Confirm that the bot has detected the tag on the conda-forge feedstock repository [conda-forge/scikit-learn-feedstock](https://github.com/conda-forge/scikit-learn-feedstock). If not, submit a PR for the release, targeting the `main` branch.

- Trigger the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml) again, but this time to upload the artifacts to the real [https://pypi.org/](https://pypi.org/). To do so, replace `testpypi` with `pypi` in the “Run workflow” form.

  **Alternatively**, it is possible to collect locally the generated binary wheel packages and source tarball and upload them all to PyPI.

  Uploading artifacts from local

  Check out at the release tag and run the following commands.

  ```
  rm -r dist
  python -m pip install -U wheelhouse_uploader twine
  python -m wheelhouse_uploader fetch \
    --version 1.9.0 --local-folder dist scikit-learn \
    https://pypi.anaconda.org/scikit-learn-wheels-staging/simple/scikit-learn/
  ```

  These commands will download all the binary packages accumulated in the [staging area on the anaconda.org hosting service](https://anaconda.org/scikit-learn-wheels-staging/scikit-learn/files) and put them in your local `./dist` folder. Check the contents of the `./dist` folder: it should contain all the wheels along with the source tarball `.tar.gz`. Make sure you do not have developer versions or older versions of the scikit-learn package in that folder. Before uploading to PyPI, you can test uploading to `test.pypi.org` first.

  ```
  twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
  ```

  Then upload everything at once to `pypi.org`.

  ```
  twine upload dist/*
  ```

- In the `main` branch, edit `doc/templates/index.html` to change the “News” section in the landing page, along with the month of the release. Do not forget to remove old entries (two years or three releases ago) and update the “On-going development” entry. Then cherry-pick it in the release branch.

- The `scikit-learn/scikit-learn.github.io` needs to be cleaned up so that ideally it stays \<5GB in size. Before doing this, create a new fresh fork of the existing repo in your own user, to have a place with the history of the repo in case it’s needed. These commands will purge the history from the repo.

  ```
  # need a non-shallow copy, and using https is much faster than ssh here
  # note that this will be a large download size, up to 100GB (repo size limit)
  git clone https://github.com/scikit-learn/scikit-learn.github.io.git
  cd scikit-learn.github.io
  git remote add write git@github.com:scikit-learn/scikit-learn.github.io.git
  # checkout an orphan branch w/o history
  git checkout --orphan temp_branch
  git add -A
  git commit -m "Initial commit after purging history"
  git branch -D main
  # rename current branch to main to replace it
  git branch -m main
  git push --force write main
  ```

- Update the symlink for `stable` and the `latestStable` variable in `versionwarning.js` in [scikit-learn/scikit-learn.github.io](https://github.com/scikit-learn/scikit-learn.github.io).

  ```
  cd /tmp
  git clone --depth 1 --no-checkout https://github.com/scikit-learn/scikit-learn.github.io.git
  cd scikit-learn.github.io
  echo stable > .git/info/sparse-checkout
  git checkout main
  rm stable
  ln -s 1.9 stable
  sed -i "s/latestStable = '.*/latestStable = '1.9';/" versionwarning.js
  git add stable versionwarning.js
  git commit -m "Update stable to point to 1.9"
  git push origin main
  ```

- Publish the release at [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn/releases) and announce it on the mailing list and social networks. Remember to add a link to the changelog in the release note. Ideally, only perform this step once the package is available both on PyPI and conda-forge and once the website is up to date.

- Update `SECURITY.md` to reflect the latest supported version `1.9.0`.

Bug-fix

Suppose that we are preparing the release `1.8.1`.

- Create a new branch from the `main` branch, then start an interactive rebase from `1.8.X` to select the commits that need to be backported:

  ```
  git rebase -i upstream/1.8.X
  ```

  This will open an interactive rebase with the `git-rebase-todo` containing all the latest commits on `main`. At this stage, you have to perform this interactive rebase with at least someone else (to not forget something and to avoid doubts).

  - Do not remove lines but drop commit by replacing `pick` with `drop`.

  - Commits to pick for a bug-fix release are *generally* prefixed with `FIX`, `CI`, and `DOC`. They should at least include all the commits of the merged PRs that were milestoned for this release.

  - Commits to `drop` for a bug-fix release are *generally* prefixed with `FEAT`, `MAINT`, `ENH`, and `API`. Reasons for not including them are to prevent change of behavior (which should only happen in major/minor releases).

  - After having dropped or picked commits, **do not exit** but paste the content of the `git-rebase-todo` message in the PR. This file is located at `.git/rebase-merge/git-rebase-todo`.

  - Save and exit to start the interactive rebase. Resolve merge conflicts when necessary.

- Create a PR targeting the `1.8.X` branch. Copy the following release checklist to the description of this PR to track the progress.

  ```
  * [ ] Set the version number in the release branch
  * [ ] Set an upper bound on build dependencies in the release branch
  * [ ] Generate the changelog in the release branch
  * [ ] Check that the wheels for the release can be built successfully
  * [ ] Merge the PR with `[cd build]` commit message to upload wheels to the staging repo
  * [ ] Upload the wheels and source tarball to https://test.pypi.org
  * [ ] Create tag on the main repo
  * [ ] Confirm bot detected at https://github.com/conda-forge/scikit-learn-feedstock
        and wait for merge
  * [ ] Upload the wheels and source tarball to PyPI
  * [ ] Update news and what's new date in main branch
  * [ ] Backport news and what's new date in release branch
  * [ ] Publish to https://github.com/scikit-learn/scikit-learn/releases
  * [ ] Announce on mailing list and on social media platforms (LinkedIn, Bluesky, etc.)
  * [ ] Update SECURITY.md in main branch
  ```

- In the release branch, change the version number `__version__` in `sklearn/__init__.py` to `1.8.1`.

- Still in the release branch, set or update the upper bound on the build dependencies in the `[build-system]` section of `pyproject.toml`. The goal is to prevent future backward incompatible releases of the dependencies to break the build in the maintenance branch.

  The upper bounds should match the latest already-released minor versions of the dependencies and should allow future micro (bug-fix) versions. For instance, if numpy 2.2.5 is the most recent version, its upper bound should be set to \<2.3.0.

- In the release branch, generate the changelog for the incoming version, i.e., `doc/whats_new/1.8.rst`. For a non RC release, push a commit where you:

  - Generate the changelog, not keeping the fragments.

    ```
    towncrier build --version 1.8.1
    ```

  - Add the list of contributor names. Suppose that the tag of the last release in the previous major/minor version is `1.9.0`, then you can use the following command to retrieve the list of contributor names:

    ```
    git shortlog -s 1.9.0.. |
      cut -f2- |
      sort --ignore-case |
      tr "\n" ";" |
      sed "s/;/, /g;s/, $//" |
      fold -s
    ```

  Then create a PR targeting the `main` branch and cherry-pick this commit there.

- Trigger the wheel builder with the `[cd build]` commit marker. See also the [workflow runs of the wheel builder](https://github.com/scikit-learn/scikit-learn/actions/workflows/wheels.yml).

  ```
  git commit --allow-empty -m "[cd build] Trigger wheel builder workflow"
  ```

  Note

  The acronym CD in `[cd build]` stands for [Continuous Delivery](https://en.wikipedia.org/wiki/Continuous_delivery) and refers to the automation used to generate the release artifacts (binary and source packages). This can be seen as an extension to CI which stands for [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration). The CD workflow on GitHub Actions is also used to automatically create nightly builds and publish packages for the development branch of scikit-learn. See also [Installing nightly builds](https://scikit-learn.org/stable/install.html#install-nightly-builds).

- Once all the CD jobs have completed successfully in the PR, merge it with the `[cd build]` marker in the commit message. This time the results will be uploaded to the staging area. You should then be able to upload the generated artifacts (`.tar.gz` and `.whl` files) to [https://test.pypi.org/](https://test.pypi.org/) using the “Run workflow” form for the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml).

  Warning

  This PR should be merged with the rebase mode instead of the usual squash mode because we want to keep the history in the `1.8.X` branch close to the history of the main branch which will help for future bug fix releases.

  In addition if on merging, the last commit, containing the `[cd build]` marker, is empty, the CD jobs won’t be triggered. In this case, you can directly push a commit with the marker in the `1.8.X` branch to trigger them.

- If the steps above went fine, proceed **with caution** to create a new tag for the release. This should be done only when you are almost certain that the release is ready, since adding a new tag to the main repository can trigger certain automated processes.

  ```
  git tag -a 1.8.1  # in the 1.8.X branch
  git push https://github.com/scikit-learn/scikit-learn.git 1.8.1
  ```

  Warning

  Don’t use the github interface for publishing the release as a way to create the tag because it will automatically send notifications to all users that follow the repo even though the website isn’t updated and wheels aren’t uploaded yet.

- Confirm that the bot has detected the tag on the conda-forge feedstock repository [conda-forge/scikit-learn-feedstock](https://github.com/conda-forge/scikit-learn-feedstock). If not, submit a PR for the release, targeting the `main` branch.

- Trigger the [PyPI publishing workflow](https://github.com/scikit-learn/scikit-learn/actions/workflows/publish_pypi.yml) again, but this time to upload the artifacts to the real [https://pypi.org/](https://pypi.org/). To do so, replace `testpypi` with `pypi` in the “Run workflow” form.

  **Alternatively**, it is possible to collect locally the generated binary wheel packages and source tarball and upload them all to PyPI.

  Uploading artifacts from local

  Check out at the release tag and run the following commands.

  ```
  rm -r dist
  python -m pip install -U wheelhouse_uploader twine
  python -m wheelhouse_uploader fetch \
    --version 1.8.1 --local-folder dist scikit-learn \
    https://pypi.anaconda.org/scikit-learn-wheels-staging/simple/scikit-learn/
  ```

  These commands will download all the binary packages accumulated in the [staging area on the anaconda.org hosting service](https://anaconda.org/scikit-learn-wheels-staging/scikit-learn/files) and put them in your local `./dist` folder. Check the contents of the `./dist` folder: it should contain all the wheels along with the source tarball `.tar.gz`. Make sure you do not have developer versions or older versions of the scikit-learn package in that folder. Before uploading to PyPI, you can test uploading to `test.pypi.org` first.

  ```
  twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
  ```

  Then upload everything at once to `pypi.org`.

  ```
  twine upload dist/*
  ```

- In the `main` branch, edit `doc/templates/index.html` to change the “News” section in the landing page, along with the month of the release. Then cherry-pick it in the release branch.

- Publish the release at [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn/releases) and announce it on the mailing list and social networks. Remember to add a link to the changelog in the release note. Ideally, only perform this step once the package is available both on PyPI and conda-forge and once the website is up to date.

- Update `SECURITY.md` to reflect the latest supported version `1.8.1`.

### Updating Authors List

This section is about updating [The people behind scikit-learn](https://scikit-learn.org/stable/about.html#authors). First create a [classic token on GitHub](https://github.com/settings/tokens/new) with the `read:org` permission. Then run the following script and enter the token when prompted:

```
cd build_tools
make authors  # Enter the token when prompted
```

### Guideline for bumping minimum versions of our dependencies

- **minimum Python version**: at the time of a minor scikit-learn release (`X.Y.0`), we drop the Python version with an initial release date of more than 4 years ago. In other words, our minimum Python version is between 3 and 4 years old.

- **compiled dependencies** (numpy, scipy, as well as compiled optional dependencies (pandas, matplotlib, pyamg, pillow, …): we take the oldest minor release (`X.Y.0`) that has wheels for our minimum Python version. In practice this means that our minimum supported version is around 3 years old, maybe a bit less.

- **pure Python dependencies** (joblib, narwhals, threadpoolctl): at the time of the scikit-learn release our minimum supported version is the most recent minor release (`X.Y.0`) that is at least 2 years old.

- we may decide to be less conservative than this guideline in some edge cases. These edge cases include: a security bugfix in one of our dependencies or a critical bugfix in one of our dependencies makes it too costly to support it in terms of maintenance.

`maint_tools/bump-dependencies-versions.py` implements these rules and can be used to give the new minimum dependency versions. It takes as input the expected scikit-learn release date, for example:

```
python maint_tools/bump-dependencies-versions.py 2025-12-01
```

### Merging Pull Requests

Individual commits are squashed when a PR is merged on GitHub. Before merging:

- The resulting commit title can be edited if necessary. Note that this will rename the PR title by default.

- The detailed description, containing the titles of all the commits, can be edited or deleted.

- For PRs with multiple code contributors, care must be taken to keep the `Co-authored-by: name <name@example.com>` tags in the detailed description. This will mark the PR as having [multiple co-authors](https://help.github.com/en/github/committing-changes-to-your-project/creating-a-commit-with-multiple-authors). Whether code contributions are significantly enough to merit co-authorship is left to the maintainer’s discretion, same as for the what’s new entry.

### The `scikit-learn.org` Website

The scikit-learn website ([https://scikit-learn.org](https://scikit-learn.org)) is hosted on GitHub, but should rarely be updated manually by pushing to the [scikit-learn/scikit-learn.github.io](https://github.com/scikit-learn/scikit-learn.github.io) repository. Most updates can be made by pushing to `main` (for `/dev`) or a release branch `A.B.X`, from which Circle CI builds and uploads the documentation automatically.

### Experimental Features

The [`sklearn.experimental`](https://scikit-learn.org/stable/api/sklearn.experimental.html#module-sklearn.experimental "sklearn.experimental") module was introduced in 0.21 and contains experimental features and estimators that are subject to change without deprecation cycle.

To create an experimental module, refer to the contents of [enable_halving_search_cv.py](https://github.com/scikit-learn/scikit-learn/blob/362cb92bb2f5b878229ea4f59519ad31c2fcee76/sklearn/experimental/enable_halving_search_cv.py), or [enable_iterative_imputer.py](https://github.com/scikit-learn/scikit-learn/blob/c9c89cfc85dd8dfefd7921c16c87327d03140a06/sklearn/experimental/enable_iterative_imputer.py).

Note

These are permalinks as in 0.24, where these estimators are still experimental. They might be stable at the time of reading, hence the permalink. See below for instructions on the transition from experimental to stable.

Note that the public import path must be to a public subpackage (like `sklearn/ensemble` or `sklearn/impute`), not just a `.py` module. Also, the (private) experimental features that are imported must be in a submodule/subpackage of the public subpackage, e.g. `sklearn/ensemble/_hist_gradient_boosting/` or `sklearn/impute/_iterative.py`. This is needed so that pickles still work in the future when the features aren’t experimental anymore.

To avoid type checker (e.g. `mypy`) errors a direct import of experimental estimators should be done in the parent module, protected by the `if typing.TYPE_CHECKING` check. See [sklearn/ensemble/\_\_init\_\_.py](https://github.com/scikit-learn/scikit-learn/blob/c9c89cfc85dd8dfefd7921c16c87327d03140a06/sklearn/ensemble/__init__.py), or [sklearn/impute/\_\_init\_\_.py](https://github.com/scikit-learn/scikit-learn/blob/c9c89cfc85dd8dfefd7921c16c87327d03140a06/sklearn/impute/__init__.py) for an example. Please also write basic tests following those in [test_enable_hist_gradient_boosting.py](https://github.com/scikit-learn/scikit-learn/blob/c9c89cfc85dd8dfefd7921c16c87327d03140a06/sklearn/experimental/tests/test_enable_hist_gradient_boosting.py).

Make sure every user-facing code you write explicitly mentions that the feature is experimental, and add a `# noqa` comment to avoid PEP8-related warnings:

```
# To use this experimental feature, we need to explicitly ask for it
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
```

For the docs to render properly, please also import `enable_my_experimental_feature` in `doc/conf.py`, otherwise sphinx will not be able to detect and import the corresponding modules. Note that using `from sklearn.experimental import *` **does not work**.

Note

Some experimental classes and functions may not be included in the [`sklearn.experimental`](https://scikit-learn.org/stable/api/sklearn.experimental.html#module-sklearn.experimental "sklearn.experimental") module, e.g., `sklearn.datasets.fetch_openml`.

Once the feature becomes stable, remove all occurrences of `enable_my_experimental_feature` in the scikit-learn code base and make the `enable_my_experimental_feature` a no-op that just raises a warning, as in [enable_hist_gradient_boosting.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/experimental/enable_hist_gradient_boosting.py). The file should stay there indefinitely as we do not want to break users’ code; we just incentivize them to remove that import with the warning. Also remember to update the tests accordingly, see [test_enable_hist_gradient_boosting.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/experimental/tests/test_enable_hist_gradient_boosting.py).

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/plotting.html>

## Developing with the Plotting API

Scikit-learn defines a simple API for creating visualizations for machine learning. The key features of this API are to run calculations once and to have the flexibility to adjust the visualizations after the fact. This section is intended for developers who wish to develop or maintain plotting tools. For usage, users should refer to the [User Guide](https://scikit-learn.org/stable/visualizations.html#visualizations).

### Plotting API Overview

This logic is encapsulated into a display object where the computed data is stored and the plotting is done in a `plot` method. The display object’s `__init__` method contains only the data needed to create the visualization. The `plot` method takes in parameters that only have to do with visualization, such as a matplotlib axes. The `plot` method will store the matplotlib artists as attributes allowing for style adjustments through the display object. The `Display` class should define one or both class methods: `from_estimator` and `from_predictions`. These methods allow creating the `Display` object from the estimator and some data or from the true and predicted values. After these class methods create the display object with the computed values, then call the display’s plot method. Note that the `plot` method defines attributes related to matplotlib, such as the line artist. This allows for customizations after calling the `plot` method.

For example, the `RocCurveDisplay` defines the following methods and attributes:

```
class RocCurveDisplay:
    def __init__(self, fpr, tpr, roc_auc, estimator_name):
        ...
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_name = estimator_name

    @classmethod
    def from_estimator(cls, estimator, X, y):
        # get the predictions
        y_pred = estimator.predict_proba(X)[:, 1]
        return cls.from_predictions(y, y_pred, estimator.__class__.__name__)

    @classmethod
    def from_predictions(cls, y, y_pred, estimator_name):
        # do ROC computation from y and y_pred
        fpr, tpr, roc_auc = ...
        viz = RocCurveDisplay(fpr, tpr, roc_auc, estimator_name)
        return viz.plot()

    def plot(self, ax=None, name=None, **kwargs):
        ...
        self.line_ = ...
        self.ax_ = ax
        self.figure_ = ax.figure_
```

Read more in [ROC Curve with Visualization API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-roc-curve-visualization-api-py) and the [User Guide](https://scikit-learn.org/stable/visualizations.html#visualizations).

### Plotting with Multiple Axes

Some of the plotting tools like [`from_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay.from_estimator "sklearn.inspection.PartialDependenceDisplay.from_estimator") and [`PartialDependenceDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay "sklearn.inspection.PartialDependenceDisplay") support plotting on multiple axes. Two different scenarios are supported:

1\. If a list of axes is passed in, `plot` will check if the number of axes is consistent with the number of axes it expects and then draws on those axes. 2. If a single axes is passed in, that axes defines a space for multiple axes to be placed. In this case, we suggest using matplotlib’s `~matplotlib.gridspec.GridSpecFromSubplotSpec` to split up the space:

```
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

fig, ax = plt.subplots()
gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

ax_top_left = fig.add_subplot(gs[0, 0])
ax_top_right = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])
```

By default, the `ax` keyword in `plot` is `None`. In this case, the single axes is created and the gridspec api is used to create the regions to plot in.

See for example, [`from_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay.from_estimator "sklearn.inspection.PartialDependenceDisplay.from_estimator") which plots multiple lines and contours using this API. The axes defining the bounding box are saved in a `bounding_ax_` attribute. The individual axes created are stored in an `axes_` ndarray, corresponding to the axes position on the grid. Positions that are not used are set to `None`. Furthermore, the matplotlib Artists are stored in `lines_` and `contours_` where the key is the position on the grid. When a list of axes is passed in, the `axes_`, `lines_`, and `contours_` are a 1d ndarray corresponding to the list of axes passed in.

### Using `matplotlib`

To keep `scikit-learn` as lightweight as possible, `matplotlib` is not a required dependency for building and using the package (it is only required for building the docs). Therefore, it is also not imported globally in the display classes, but only within the plotting functions where it is actually needed. Before importing it, use `check_matplotlib_support` from [\_optional_dependencies.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_optional_dependencies.py). This will check if it is installed, and if not, also raises a comprehensive error message including the caller that requested it, for reference.

For testing, use the `pyplot` fixture from [conftest.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/conftest.py) as the first argument in every test that requires it. This imports `matplotlib.pyplot` (or skips the test, if it is not installed) and also takes care of closing all figures before and after running the test.

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/callbacks.html>

## Developing with the callback API

Scikit-learn offers a [callback](https://scikit-learn.org/stable/glossary.html#term-callback) API to use built-in or custom callbacks with compatible estimators. This section is intended for developers who wish to implement callbacks or add callback support in estimators. For a general introduction to callbacks and how to use them, see the [user guide](https://scikit-learn.org/stable/callbacks.html#callbacks-user).

- [Implementing callback support in estimators](https://scikit-learn.org/stable/developers/callback_support.html)
  - [The CallbackSupportMixin class](https://scikit-learn.org/stable/developers/callback_support.html#the-callbacksupportmixin-class)
  - [The CallbackContext class](https://scikit-learn.org/stable/developers/callback_support.html#the-callbackcontext-class)
  - [The with_callbacks decorator](https://scikit-learn.org/stable/developers/callback_support.html#the-with-callbacks-decorator)
  - [Minimal example](https://scikit-learn.org/stable/developers/callback_support.html#minimal-example)
- [Developing callbacks](https://scikit-learn.org/stable/developers/developing_callbacks.html)
  - [The callback protocol](https://scikit-learn.org/stable/developers/developing_callbacks.html#the-callback-protocol)
  - [Auto-propagated callbacks](https://scikit-learn.org/stable/developers/developing_callbacks.html#auto-propagated-callbacks)
  - [Callback shared state](https://scikit-learn.org/stable/developers/developing_callbacks.html#callback-shared-state)
  - [Minimal example](https://scikit-learn.org/stable/developers/developing_callbacks.html#minimal-example)

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/callback_support.html>

## Implementing callback support in estimators

Adding [callback](https://scikit-learn.org/stable/glossary.html#term-callback) support in an estimator boils down to enabling the registration of callbacks, expressing [fit](https://scikit-learn.org/stable/glossary.html#term-fit) as a tree of [tasks](https://scikit-learn.org/stable/glossary.html#term-fit-task), and invoking the callbacks at the beginning and end of each of these tasks. To achieve this, scikit-learn provides the following helpers from the [`callback`](https://scikit-learn.org/stable/api/sklearn.callback.html#module-sklearn.callback "sklearn.callback") module:

- [`CallbackSupportMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin "sklearn.callback.CallbackSupportMixin"), which enables callback registration and initializes callback handling at the beginning of fit.

- [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext"), which represents tasks and is the central object for managing callbacks during fit.

- [`with_callbacks`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.with_callbacks.html#sklearn.callback.with_callbacks "sklearn.callback.with_callbacks"), to guarantee proper callback teardown at the end of fit.

### The CallbackSupportMixin class

To support callbacks, an estimator must inherit from the [`CallbackSupportMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin "sklearn.callback.CallbackSupportMixin") class, which exposes the following methods:

- [`set_callbacks`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin.set_callbacks "sklearn.callback.CallbackSupportMixin.set_callbacks"), a public method to be called by the user to register callbacks on the estimator.

- [`_init_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin._init_callback_context "sklearn.callback.CallbackSupportMixin._init_callback_context"), which should be called at the beginning of fit to create the root [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext"), corresponding to the task that represents the entire execution of `fit`. This method also sets up the callbacks that are registered on the estimator.

  Note

  While the leading underscore signals that [`_init_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin._init_callback_context "sklearn.callback.CallbackSupportMixin._init_callback_context") is intended for internal use and should not appear in auto-completion suggestions for end users, it is made available to developers building third-party estimators and should be considered part of the public API contract.

### The CallbackContext class

The [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext") objects are responsible for invoking the callbacks at the right time during fit. They track the different tasks of the estimator, with one context instance representing each task, and capture the tree structure of the tasks involved in the execution of the fit method.

A task is an arbitrary unit of work defined by the estimator. Usually, a task corresponds to an iteration of the estimator’s learning algorithm. They can also correspond to steps of a pipeline, cross-validation folds, etc. As tasks can be decomposed into subtasks, the tasks (and therefore callback contexts) have a natural tree structure, with the root task being the whole fit task.

The callback context objects follow this tree structure, holding references to their parent and children contexts, and are dynamically built during `fit`. The root context must be created by the [`_init_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin._init_callback_context "sklearn.callback.CallbackSupportMixin._init_callback_context") method.

examples of task / context trees

As an example, [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans "sklearn.cluster.KMeans") has two nested loops: the outer loop is controlled by the `n_init` parameter, and the inner loop is controlled by the `max_iter` parameter. Therefore its task tree looks like this:

```
KMeans fit (root)
├── init 0
│   ├── iter 0
│   ├── iter 1
│   ├── ...
│   └── iter n
├── init 1
│   ├── iter 0
│   ├── ...
│   └── iter n
└── init 2
    ├── iter 0
    ├── ...
    └── iter n
```

where each innermost `iter j` task corresponds to the computation of the labels and centers for the full dataset. A callback registered on a KMeans estimator thus will be invoked at the beginning and end of the `fit` task, each of the outer `init i` tasks and each of the inner `iter j` tasks.

By convention, for performance reasons and consistency across estimators, the innermost tasks of scikit-learn estimators, i.e. the leaves of the task tree, correspond to operations on the full input data (or batches for incremental estimators).

When the estimator is a meta-estimator, a task leaf usually corresponds to fitting a sub-estimator. Therefore, this leaf and the root task of the sub-estimator actually represent the same task. In this case the leaf task of the meta-estimator and the root task of the sub-estimator are merged into a single task. The task trees of the meta-estimator and the sub-estimator are combined into a single task tree. For instance, a [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline") would have a task tree that looks like this:

```
Pipeline fit (root)
├── step 0 | StandardScaler fit
│   └── <insert StandardScaler task tree here>
└── step 1 | LogisticRegression fit
    └── <insert LogisticRegression task tree here>
```

To dynamically build the context tree and manage the callbacks during fit, the [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext") class exposes the following methods:

- [`subcontext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext.subcontext "sklearn.callback.CallbackContext.subcontext")

  This method should be used to create a context for a subtask. Callback contexts must not be created directly but through this method (or [`_init_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackSupportMixin.html#sklearn.callback.CallbackSupportMixin._init_callback_context "sklearn.callback.CallbackSupportMixin._init_callback_context") for the root context).

- [`call_on_fit_task_begin`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext.call_on_fit_task_begin "sklearn.callback.CallbackContext.call_on_fit_task_begin") and [`call_on_fit_task_end`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext.call_on_fit_task_end "sklearn.callback.CallbackContext.call_on_fit_task_end")

  ```
  def call_on_fit_task_begin(
      self, *, estimator, X=None, y=None, metadata=None, reconstruction_attributes=None
  ) -> None: ...

  def call_on_fit_task_end(
      self, *, estimator, X=None, y=None, metadata=None, reconstruction_attributes=None
  ) -> bool: ...
  ```

  These two methods must be called respectively at the beginning and end of the task that the context is responsible for. As their name suggests, they call the `on_fit_task_begin` and `on_fit_task_end` methods of the callbacks registered on the estimator.

  In addition to the callback context that is implicitly passed to the registered callbacks, the keyword arguments of `call_on_fit_task_begin/end` are used to pass additional information about the state of the fitting process at a given task. It is not expected to provide a value for all of them at every call of these methods. Estimators are expected to provide all the values that they are capable to produce. Callbacks then adapt their behavior based on the provided values for a given task.

  The `reconstruction_attributes` kwarg

  When `call_on_fit_task_begin/end` is called, the state of the estimator at this task is likely to be incomplete and thus unable to [predict](https://scikit-learn.org/stable/glossary.html#term-predict), [transform](https://scikit-learn.org/stable/glossary.html#term-transform), etc … The `reconstruction_attributes` kwarg expects a dictionary containing the necessary missing attributes to set on the estimator to ensure that it is ready to [predict](https://scikit-learn.org/stable/glossary.html#term-predict), [transform](https://scikit-learn.org/stable/glossary.html#term-transform), etc … as if fit had stopped at this task.

  The callback context will copy the state of the estimator at this task, set the reconstruction attributes and pass the resulting estimator to the callbacks as `fitted_estimator`.

  If no additional attributes are needed to make the estimator ready, an empty dictionary should be passed instead of leaving the default value otherwise the callback context won’t pass a `fitted_estimator` to the callbacks.

  Lazy evaluation of the kwargs

  For each of these kwargs, a callable (with no arguments and returning the kwarg value) can be provided instead of the actual value. When it is the case, if a callback requires the kwarg, the callback context will evaluate the callable and forward the returned value to the callback. This mechanism enables lazy evaluation of the kwarg values, to avoid potentially costly computations when no callback requires a kwarg value.

  To prevent performance degradations, estimators should lazily pass quantities that are expensive to compute.

  Interrupting `fit`

  The `call_on_fit_task_end` method returns a boolean, which can be used to interrupt the current level of iterations, to implement early stopping for instance. It returns `True` if any callback signaled to stop the `fit` process at the end of this task and `False` otherwise.

- [`propagate_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext.propagate_callback_context "sklearn.callback.CallbackContext.propagate_callback_context").

  This method enables combining the context trees of individual estimators and meta-estimators in estimator compositions (e.g. a `GridSearchCV` on a `LogisticRegression`) into a single context tree, rooted at the fit of the top level estimator.

  It should be used in a meta-estimator, on a context corresponding to the task of fitting a sub-estimator. This task is both a leaf task of the meta-estimator and the root task of the sub-estimator. Their corresponding contexts are thus merged into a single context in the combined tree.

  In addition, [`propagate_callback_context`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext.propagate_callback_context "sklearn.callback.CallbackContext.propagate_callback_context") is a context manager that propagates the [auto-propagated](https://scikit-learn.org/stable/glossary.html#term-auto-propagated) callbacks from the meta-estimator to the sub-estimator such that they are called at the tasks of the sub-estimator as well. It also clears the propagated callbacks on exit such that the fitted sub-estimator no longer holds any locally registered callbacks.

### The with_callbacks decorator

For third-party estimators implementing callback support, the `fit` method should be decorated with the [`with_callbacks`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.with_callbacks.html#sklearn.callback.with_callbacks "sklearn.callback.with_callbacks") decorator. This decorator guarantees that the callbacks are torn down after `fit` finishes, even if it exits on an error.

For scikit-learn’s built-in estimators, the `_fit_context` decorator already takes care of the callbacks teardown, thus [`with_callbacks`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.with_callbacks.html#sklearn.callback.with_callbacks "sklearn.callback.with_callbacks") should not be used.

### Minimal example

Here is a typical implementation of callback support in a custom estimator:

```
from sklearn.callback import CallbackSupportMixin, with_callbacks


class MyEstimator(CallbackSupportMixin):
    def __init__(self, max_iter):
        self.max_iter = max_iter

    @with_callbacks
    def fit(self, X, y):
        callback_ctx = self._init_callback_context(max_subtasks=self.max_iter)
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext(task_name="iteration")
            subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)

            # Do something

            if subcontext.call_on_fit_task_end(estimator=self, X=X, y=y):
                break

        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        return self
```

______________________________________________________________________

**Original page:** <https://scikit-learn.org/stable/developers/developing_callbacks.html>

## Developing callbacks

### The callback protocol

To be compatible with scikit-learn estimators, [callbacks](https://scikit-learn.org/stable/glossary.html#term-callbacks) must implement the [`FitCallback`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback "sklearn.callback.FitCallback") [protocol](https://typing.python.org/en/latest/spec/protocol.html):

```
class FitCallback(Protocol):

    def setup(self, estimator, context) -> None: ...

    def on_fit_task_begin(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None
    ) -> None: ...

    def on_fit_task_end(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None
    ) -> bool: ...

    def teardown(self, estimator, context) -> None: ...
```

The methods of the protocol, referred to as callback [hooks](https://scikit-learn.org/stable/glossary.html#term-hooks), will be called at specific steps during the fitting process of the estimator the callback is registered on:

- [`setup`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.setup "sklearn.callback.FitCallback.setup") and [`teardown`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.teardown "sklearn.callback.FitCallback.teardown")

  These hooks are only called once, respectively at the start and end of `fit`. They take care of setting up and tearing down the callback, like allocating and freeing resources for instance.

- [`on_fit_task_begin`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.on_fit_task_begin "sklearn.callback.FitCallback.on_fit_task_begin") and [`on_fit_task_end`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.on_fit_task_end "sklearn.callback.FitCallback.on_fit_task_end")

  These hooks are called at the beginning and end of each [task](https://scikit-learn.org/stable/developers/callback_support.html#callback-task-definition) during `fit`.

  In concrete implementations of callbacks, only the optional keyword-only arguments actually used by the hook should be explicitly declared in the hook signature. The presence of an argument in the signature signals that the hook requires that argument, which allows the callback framework to avoid computing values that are not used by any registered callback.

  Warning

  These arguments must be defined as **keyword only**. If the kwargs are not keyword only, the values will not be provided to the hooks.

  Even if requested, the optional arguments might or might not be provided by the estimator, depending on its ability to produce them at this task. Thus the implementation of the hooks should not expect to always receive a value for each of them and adapt their behavior accordingly.

  Interrupting `fit`

  The [`on_fit_task_end`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.on_fit_task_end "sklearn.callback.FitCallback.on_fit_task_end") hook returns a boolean, which when set to `True`, requests the estimator to stop the `fit` process at this task. Note that estimators that don’t aim to be interruptible will ignore this request and continue with the next task.

All the hooks receive, as mandatory arguments, the estimator instance calling the callback and the [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext") object holding the contextual information that allows unique identification of the task that is being processed as public attributes. See [`CallbackContext`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.CallbackContext.html#sklearn.callback.CallbackContext "sklearn.callback.CallbackContext") for more details.

The `estimator` argument

The estimator instance received by the hooks, as a mandatory argument, is in the same state as it was when calling the hook during `fit`. Therefore it is not expected to be fully fitted (except for the [`teardown`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.teardown "sklearn.callback.FitCallback.teardown") hook). Callbacks should not rely on it to [predict](https://scikit-learn.org/stable/glossary.html#term-predict), [transform](https://scikit-learn.org/stable/glossary.html#term-transform), etc … but rather use the `fitted_estimator` when available.

### Auto-propagated callbacks

[Auto-propagated](https://scikit-learn.org/stable/glossary.html#term-auto-propagated) callbacks, i.e. callbacks that are expected to be propagated from meta-estimators to their sub-estimators, must implement the [`AutoPropagatedCallback`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.AutoPropagatedCallback.html#sklearn.callback.AutoPropagatedCallback "sklearn.callback.AutoPropagatedCallback") protocol, an extension of the [`FitCallback`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback "sklearn.callback.FitCallback") protocol:

```
class AutoPropagatedCallback(FitCallback, Protocol):

    @property
    def max_propagation_depth(self) -> int | None: ...
```

By contrast with regular callbacks that are only invoked at the tasks of the estimator on which they are registered, auto-propagated callbacks are invoked at the tasks of all the estimators in estimator compositions, up to the maximum propagation depth. If set to 0, the callback is not propagated to sub-estimators and only invoked at the tasks of the top-level estimator. If set to `None`, the callback is propagated to sub-estimators at all nesting levels.

Auto-propagated callbacks should be registered on the top-level estimator. If the top-level estimator does not support callbacks, they can be registered on sub-estimators and are expected to work, though possibly not at full capacity.

Note

The [`setup`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.setup "sklearn.callback.FitCallback.setup") and [`teardown`](https://scikit-learn.org/stable/modules/generated/sklearn.callback.FitCallback.html#sklearn.callback.FitCallback.teardown "sklearn.callback.FitCallback.teardown") hooks of an auto-propagated callback are also called only once, at the beginning and end of the top-level estimator’s `fit` method. They are not called for any of its sub-estimators.

### Callback shared state

Since the estimator on which the callback is registered may be cloned and fitted multiple times in a meta-estimator, callbacks should behave as if the same callback instance were registered on multiple estimators. Therefore, `setup` / `teardown` should not reset the state of the callback, and `on_fit_task_begin` / `on_fit_task_end` should accumulate data across all fits. Indeed resetting state in `setup` / `teardown` would drop information collected from previous or concurrent fits.

### Minimal example

Here is an example implementation of a simple custom callback that prints a message every time it is invoked:

```
class MyCallback:

    def setup(self, estimator, context):
        print(f"Setup hook is being called in the {context.task_name} task.")

    def teardown(self, estimator, context):
        print(f"Teardown hook is being called in the {context.task_name} task.")

    def on_fit_task_begin(self, estimator, context, *, X=None):
        msg = f"{context.task_name} task is starting."
        if X is not None:
            msg += f" With training data of shape {X.shape}."
        print(msg)

    def on_fit_task_end(
        self, estimator, context, *, X=None, y=None, fitted_estimator=None
    ):
        msg = f"{context.task_name} task is ending."
        mean_squared_error = ((y - fitted_estimator.predict(X))**2).mean()
        msg += f" With a mean squared error of {mean_squared_error}."
        print(msg)
```
