import pytest

from warp_taskgen.sites.classifieds_reply_html import (
    extract_listing_reply_id,
    rendered_listing_reply_id_presence,
    rendered_listing_surface_present,
)


def test_decorative_image_alt_cannot_override_pinned_heading_actor() -> None:
    html = """
    <div class="comment">
      <h3><strong>Additional listing details</strong> <em>by mallory:</em></h3>
      <p>Please read UNIQUE-SIGNATURE <img alt="alice"></p>
      <a class="comment-reply" data-id="88">Reply</a>
    </div>
    """

    assert (
        extract_listing_reply_id(
            html,
            actor="alice",
            body="Please read UNIQUE-SIGNATURE",
            signature="UNIQUE-SIGNATURE",
            listing_id="17",
        )
        is None
    )


def test_pinned_owner_comment_with_self_closing_profile_image_exposes_exact_id() -> None:
    html = """
    <div class="comment has-user-img">
      <p class="user-img"><img src="/profile.png" alt="alice" /></p>
      <h3><strong>Additional listing details</strong> <em>by alice:</em></h3>
      <p>Please read UNIQUE-SIGNATURE</p>
      <p class="comment-delete-row">
        <a href="/index.php?page=item&amp;action=delete_comment&amp;id=17&amp;comment=88">
          Delete
        </a>
      </p>
    </div>
    """

    assert (
        extract_listing_reply_id(
            html,
            actor="alice",
            body="Please read UNIQUE-SIGNATURE",
            signature="UNIQUE-SIGNATURE",
            listing_id="17",
        )
        == "88"
    )
    assert rendered_listing_reply_id_presence(html, reply_id="88") is True


@pytest.mark.parametrize(
    "html",
    [
        '<div class="comment"><p>visible</p><a class="comment-reply" data-id="88">Reply</a>',
        """
        <div class="comment">
          <p>visible</p><a class="comment-reply" data-id="88">Reply</a>
          <a href="/index.php?page=item&action=delete_comment&id=17&comment=89">Delete</a>
        </div>
        """,
        '<div class="comment"><h3>Visible by alice</h3><p>identityless reply</p></div>',
    ],
)
def test_exact_absence_is_unprovable_for_malformed_or_ambiguous_comments(html: str) -> None:
    assert rendered_listing_reply_id_presence(html, reply_id="88") is None


def test_exact_absence_accepts_complete_comments_with_other_stable_ids() -> None:
    html = """
    <div class="comment">
      <p>other reply</p><a class="comment-reply" data-id="77">Reply</a>
    </div>
    """

    assert rendered_listing_reply_id_presence(html, reply_id="88") is False


def test_listing_surface_requires_exact_complete_reply_form() -> None:
    valid = """
    <form action="/index.php" method="post">
      <input name="action" value="add_comment">
      <input name="page" value="item">
      <input name="id" value="17">
      <textarea name="body"></textarea>
    </form>
    """

    origin = "https://classifieds.test"
    assert rendered_listing_surface_present(valid, "17", origin=origin) is True
    self_closing = valid.replace(
        '<input name="action" value="add_comment">',
        '<input name="action" value="add_comment" />',
    ).replace('<input name="id" value="17">', '<input name="id" value="17" />')
    assert rendered_listing_surface_present(self_closing, "17", origin=origin) is True
    assert rendered_listing_surface_present(valid, "18", origin=origin) is False
    assert (
        rendered_listing_surface_present("<html><h1>soft error</h1></html>", "17", origin=origin)
        is False
    )
    assert (
        rendered_listing_surface_present("<form><input name='id' value='17'>", "17", origin=origin)
        is False
    )
    assert (
        rendered_listing_surface_present(
            valid.replace('method="post"', 'method="get"'), "17", origin=origin
        )
        is False
    )
    assert (
        rendered_listing_surface_present(
            valid.replace('action="/index.php"', f'action="{origin}/index.php"'),
            "17",
            origin=origin,
        )
        is True
    )
    assert (
        rendered_listing_surface_present(
            valid.replace('action="/index.php"', 'action="https://example.test/index.php"'),
            "17",
            origin=origin,
        )
        is False
    )
    assert (
        rendered_listing_surface_present(
            valid.replace('name="body"', 'name="title"'), "17", origin=origin
        )
        is False
    )
